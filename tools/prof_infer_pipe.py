# Instrumented copy of InferPipe.pipeline for measurement only.
import time, sys, atexit
import infer_pipe as ip

T0 = time.monotonic()
STAGES = ["pull_tensor", "infer", "pull_frame", "post", "push", "total"]
acc = {s: [] for s in STAGES}
first = [None]

def pipeline(self):
    while self.stop_thread == False:
        t0 = time.monotonic()
        input_img = self.gst_pipe.pull_tensor(self.gst_pre_inp, self.sub_flow.input.loop,
            self.sub_flow.model.crop[0], self.sub_flow.model.crop[1],
            self.sub_flow.model.data_layout, self.sub_flow.model.input_tensor_types[0])
        if input_img is None: break
        t1 = time.monotonic()
        result = self.run_time(input_img)
        t2 = time.monotonic()
        frame = self.gst_pipe.pull_frame(self.gst_sen_inp, self.sub_flow.input.loop)
        if frame is None: break
        t3 = time.monotonic()
        out = self.post_proc(frame, result)
        t4 = time.monotonic()
        self.gst_pipe.push_frame(out, self.gst_post_out)
        t5 = time.monotonic()
        if first[0] is None:
            first[0] = t5 - T0
            print("[PROF] first frame out %.2fs after import" % first[0], flush=True)
        for k, v in zip(STAGES, [t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t5-t0]):
            acc[k].append(v * 1000)
        n = len(acc["total"])
        if n % 150 == 0:
            report()
    self.stop_thread = True
    self.gst_pipe.send_eos(self.gst_post_out)

def report():
    import statistics as st
    n = len(acc["total"])
    el = sum(acc["total"][-150:]) / 1000
    print("[PROF] frames=%d fps(last150)=%.1f" % (n, 150 / el if el else 0), flush=True)
    for k in STAGES:
        v = sorted(acc[k][-150:])
        print("[PROF]   %-11s mean=%6.2f p50=%6.2f p95=%6.2f max=%6.2f ms" % (
            k, st.mean(v), v[len(v)//2], v[int(len(v)*.95)], v[-1]), flush=True)

ip.InferPipe.pipeline = pipeline
