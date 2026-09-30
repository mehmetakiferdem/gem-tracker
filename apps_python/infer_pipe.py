#  Copyright (C) 2021 Texas Instruments Incorporated - http://www.ti.com/
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#    Neither the name of Texas Instruments Incorporated nor the names of
#    its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from time import time
import queue
import threading
import utils
import debug
from post_process import PostProcess

class InferPipe:
    """
    Class to abstract the threading of multiple inference pipelines
    """

    def __init__(self, sub_flow, gst_pipe):
        """
        Constructor to create an InferPipe object.
        Args:
            sub_flow: sub_flow configuration
            gst_pipe: gstreamer pipe object
        """
        self.sub_flow = sub_flow
        self.gst_pipe = gst_pipe
        self.gst_pre_inp = gst_pipe.get_src(sub_flow.gst_pre_src_name, sub_flow.flow.id)
        self.gst_sen_inp = gst_pipe.get_src(sub_flow.gst_sen_src_name, sub_flow.flow.id)
        self.run_time = sub_flow.model.run_time
        self.post_proc = PostProcess.get(sub_flow)

        # Sized for the frame post processing returns.
        self.gst_post_out = gst_pipe.get_sink(
            sub_flow.gst_post_sink_name,
            sub_flow.post_proc_width,
            sub_flow.post_proc_height,
            sub_flow.input.fps,
        )
        self.param = sub_flow.model
        self.pre_proc_debug = None
        self.infer_debug = None

        if sub_flow.debug_config:
            if sub_flow.debug_config.pre_proc:
                self.pre_proc_debug = debug.Debug(sub_flow.debug_config, "pre")
            if sub_flow.debug_config.inference:
                self.infer_debug = debug.Debug(sub_flow.debug_config, "infer")

        # Letterbox: the tensor comes out of the pipeline at the aspect
        # preserving size and is copied into the top left of a buffer of the
        # model input size, pre-filled with the pad colour once. With input
        # optimization off the pad value would have to be normalised too;
        # every letterboxed model in the zoo has it on.
        self.letterbox = getattr(sub_flow.model, "letterbox_size", None)
        if self.letterbox:
            m = sub_flow.model
            cw, ch = m.crop
            shape = (1, 3, ch, cw) if m.data_layout == "NCHW" else (1, ch, cw, 3)
            pad = m.pad_color
            if m.mean is not None and m.scale is not None:
                pad = [(pad - mu) * sc for mu, sc in zip(m.mean, m.scale)]
                pad = np.array(pad).reshape(
                    (1, 3, 1, 1) if m.data_layout == "NCHW" else (1, 1, 1, 3)
                )
            self.pad_buffer = np.empty(shape, m.input_tensor_types[0])
            self.pad_buffer[...] = pad
            self.pull_size = self.letterbox
        else:
            self.pull_size = sub_flow.model.crop

        self.pipeline_thread = threading.Thread(target=self.pipeline)
        self.stop_thread = False

    def start(self):
        """
        Start the pipeline
        """
        self.pipeline_thread.start()

    def stop(self):
        """
        Stop the pipeline
        """
        self.stop_thread = True

    def pipeline(self):
        """
        Callback function for pipeline thread
        """
        # An exception in here used to end this thread without setting
        # stop_thread, so wait_for_exit() never returned: the service stayed
        # "active" with a frozen picture and systemd never restarted it. Now
        # any exception still stops the demo, the process exits, and
        # Restart=always brings it back.
        try:
            self._pipeline()
        finally:
            self.stop_thread = True

    def _pipeline(self):
        if getattr(self.param, "extra_run_times", None):
            self._pipeline_parallel([self.run_time] + self.param.extra_run_times)
        else:
            self._pipeline_serial()

    def _pull(self):
        """
        Next (tensor, frame) pair, letterboxed tensor in a buffer of its own
        so several can be in flight. None at end of stream.
        """
        input_img = self.gst_pipe.pull_tensor(
            self.gst_pre_inp,
            self.sub_flow.input.loop,
            self.pull_size[0],
            self.pull_size[1],
            self.sub_flow.model.data_layout,
            self.sub_flow.model.input_tensor_types[0],
        )
        if input_img is None:
            return None
        if self.letterbox:
            w, h = self.letterbox
            buf = self.pad_buffer.copy()
            if self.sub_flow.model.data_layout == "NCHW":
                buf[:, :, :h, :w] = input_img
            else:
                buf[:, :h, :w, :] = input_img
            input_img = buf
        # The frame belongs to the same capture as the tensor; pull it right
        # away so the pair can travel through the workers together.
        frame = self.gst_pipe.pull_frame(self.gst_sen_inp, self.sub_flow.input.loop)
        if frame is None:
            return None
        return input_img, frame

    def _pipeline_parallel(self, run_times):
        """
        One capture loop (this thread), one inference worker per C7x runtime,
        results handed on strictly in capture order: tracking, the servo loop
        and the Kalman filters all assume consecutive frames.

        Capture blocks while every worker is busy, like the serial loop does
        during inference; the appsinks keep at most two buffers and drop the
        older ones, so waiting never builds up latency.
        """
        jobs = queue.Queue(maxsize=len(run_times))
        results = {}
        results_cv = threading.Condition()
        stopping = threading.Event()
        errors = []

        def worker(run_time):
            try:
                while not stopping.is_set():
                    try:
                        seq, tensor, frame = jobs.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    start = time()
                    result = run_time(tensor)
                    self.sub_flow.report.report_proctime("dl-inference", time() - start)
                    with results_cv:
                        results[seq] = (frame, result)
                        results_cv.notify_all()
            except BaseException as e:
                errors.append(e)
                stopping.set()

        def output():
            try:
                nxt = 0
                while not stopping.is_set():
                    with results_cv:
                        if nxt not in results:
                            results_cv.wait(0.2)
                            continue
                        frame, result = results.pop(nxt)
                    nxt += 1
                    if self.infer_debug:
                        self.infer_debug.log(str(result))
                    out_frame = self.post_proc(frame, result)
                    self.gst_pipe.push_frame(out_frame, self.gst_post_out)
                    self.sub_flow.report.report_frame()
            except BaseException as e:
                errors.append(e)
                stopping.set()

        threads = [threading.Thread(target=worker, args=(rt,), daemon=True)
                   for rt in run_times]
        threads.append(threading.Thread(target=output, daemon=True))
        for t in threads:
            t.start()

        seq = 0
        try:
            while not self.stop_thread and not stopping.is_set():
                pair = self._pull()
                if pair is None:
                    break
                tensor, frame = pair
                if self.pre_proc_debug:
                    self.pre_proc_debug.log(str(tensor.flatten()))
                while not stopping.is_set():
                    try:
                        jobs.put((seq, tensor, frame), timeout=0.2)
                        break
                    except queue.Full:
                        pass
                seq += 1
        finally:
            stopping.set()
            for t in threads:
                t.join(2.0)
            self.stop_thread = True
            self.gst_pipe.send_eos(self.gst_post_out)
        if errors:
            raise errors[0]

    def _pipeline_serial(self):
        while self.stop_thread == False:
            # capture and pre-process
            input_img = self.gst_pipe.pull_tensor(
                self.gst_pre_inp,
                self.sub_flow.input.loop,
                self.pull_size[0],
                self.pull_size[1],
                self.sub_flow.model.data_layout,
                self.sub_flow.model.input_tensor_types[0],
            )
            if type(input_img) == type(None):
                break

            if self.letterbox:
                w, h = self.letterbox
                if self.sub_flow.model.data_layout == "NCHW":
                    self.pad_buffer[:, :, :h, :w] = input_img
                else:
                    self.pad_buffer[:, :h, :w, :] = input_img
                input_img = self.pad_buffer

            if self.pre_proc_debug:
                self.pre_proc_debug.log(str(input_img.flatten()))

            # Inference
            start = time()
            result = self.run_time(input_img)
            end = time()
            self.sub_flow.report.report_proctime("dl-inference", (end - start))

            if self.infer_debug:
                self.infer_debug.log(str(result))

            # post-process
            frame = self.gst_pipe.pull_frame(self.gst_sen_inp, self.sub_flow.input.loop)
            if type(frame) == type(None):
                break
            out_frame = self.post_proc(frame, result)
            self.gst_pipe.push_frame(out_frame, self.gst_post_out)
            # Increment frame count
            self.sub_flow.report.report_frame()

        self.stop_thread = True
        self.gst_pipe.send_eos(self.gst_post_out)