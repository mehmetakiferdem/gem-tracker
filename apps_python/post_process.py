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

import cv2
import numpy as np
import copy
import sys
import debug
import utils
from typing import List

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# norfair and the tracking helper module path_draw are imported inside
# PostProcessTracking.__init__() instead of here on purpose: post_process is
# imported by every flow, importing norfair costs ~7s on this target, and only
# a flow with "enable_tracking: True" needs it. Classification, detection,
# segmentation and keypoint_detection flows therefore keep their original
# startup time and do not require norfair to be installed at all.


def create_title_frame(title, width, height):
    frame = np.zeros((height, width, 3), np.uint8)
    if title != None:
        frame = cv2.putText(
            frame,
            "Texas Instruments - Edge Analytics",
            (40, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            2,
        )
        frame = cv2.putText(
            frame, title, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    return frame


def overlay_model_name(frame, model_name, start_x, start_y, width, height):
    row_size = 40 * width // 1280
    font_size = width / 1280
    cv2.putText(
        frame,
        "Model : " + model_name,
        (start_x + 5, start_y - row_size // 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        2,
    )
    return frame


class PostProcess:
    """
    Class to create a post process context
    """

    def __init__(self, flow):
        self.flow = flow
        self.model = flow.model
        self.debug = None
        self.debug_str = ""
        if flow.debug_config and flow.debug_config.post_proc:
            self.debug = debug.Debug(flow.debug_config, "post")

    def get(flow):
        """
        Create a object of a subclass based on the task type
        """
        if flow.model.task_type == "classification":
            return PostProcessClassification(flow)
        elif flow.model.task_type == "detection":
            # A detection model draws plain bounding boxes unless the config
            # opts in to tracking with "enable_tracking: True".
            if getattr(flow.model, "enable_tracking", False):
                return PostProcessTracking(flow)
            return PostProcessDetection(flow)
        elif flow.model.task_type == "segmentation":
            return PostProcessSegmentation(flow)
        elif flow.model.task_type == "keypoint_detection":
            return PostProcessKeypointDetection(flow)


class PostProcessClassification(PostProcess):
    def __init__(self, flow):
        super().__init__(flow)

    def __call__(self, img, results):
        """
        Post process function for classification
        Args:
            img: Input frame
            results: output of inference
        """
        results = np.squeeze(results)
        img = self.overlay_topN_classnames(img, results)

        if self.debug:
            self.debug.log(self.debug_str)
            self.debug_str = ""

        return img

    def overlay_topN_classnames(self, frame, results):
        """
        Process the results of the image classification model and draw text
        describing top 5 detected objects on the image.

        Args:
            frame (numpy array): Input image in BGR format where the overlay should
        be drawn
            results (numpy array): Output of the model run
        """
        orig_width = frame.shape[1]
        orig_height = frame.shape[0]
        row_size = 40 * orig_width // 1280
        font_size = orig_width / 1280
        N = self.model.topN
        topN_classes = np.argsort(results)[: (-1 * N) - 1 : -1]
        title_text = "Recognized Classes (Top %d):" % N
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_size, _ = cv2.getTextSize(title_text, font, font_size, 2)

        bg_top_left = (0, (2 * row_size) - text_size[1] - 5)
        bg_bottom_right = (text_size[0] + 10, (2 * row_size) + 3 + 5)
        font_coord = (5, 2 * row_size)

        cv2.rectangle(frame, bg_top_left, bg_bottom_right, (5, 11, 120), -1)

        cv2.putText(
            frame,
            title_text,
            font_coord,
            font,
            font_size,
            (0, 255, 0),
            2,
        )
        row = 3
        for idx in topN_classes:
            idx = idx + self.model.label_offset
            if idx in self.model.dataset_info:
                class_name = self.model.dataset_info[idx].name
                if not class_name:
                    class_name = "UNDEFINED"
                if self.model.dataset_info[idx].supercategory:
                    class_name = (
                        self.model.dataset_info[idx].supercategory + "/" + class_name
                    )
            else:
                class_name = "UNDEFINED"

            text_size, _ = cv2.getTextSize(class_name, font, font_size, 2)

            bg_top_left = (0, (row_size * row) - text_size[1] - 5)
            bg_bottom_right = (text_size[0] + 10, (row_size * row) + 3 + 5)
            font_coord = (5, row_size * row)

            cv2.rectangle(frame, bg_top_left, bg_bottom_right, (5, 11, 120), -1)
            cv2.putText(
                frame,
                class_name,
                font_coord,
                font,
                font_size,
                (255, 255, 0),
                2,
            )
            row = row + 1
            if self.debug:
                self.debug_str += class_name + "\n"

        return frame


class PostProcessDetection(PostProcess):
    def __init__(self, flow):
        super().__init__(flow)

    def __call__(self, img, results):
        """
        Post process function for detection
        Args:
            img: Input frame
            results: output of inference
        """
        for i, r in enumerate(results):
            r = np.squeeze(r)
            if r.ndim == 1:
                r = np.expand_dims(r, 1)
            results[i] = r

        if self.model.shuffle_indices:
            results_reordered = []
            for i in self.model.shuffle_indices:
                results_reordered.append(results[i])
            results = results_reordered

        if results[-1].ndim < 2:
            results = results[:-1]

        bbox = np.concatenate(results, axis=-1)

        if self.model.formatter:
            if self.model.ignore_index == None:
                bbox_copy = copy.deepcopy(bbox)
            else:
                bbox_copy = copy.deepcopy(np.delete(bbox, self.model.ignore_index, 1))
            bbox[..., self.model.formatter["dst_indices"]] = bbox_copy[
                ..., self.model.formatter["src_indices"]
            ]

        if not self.model.normalized_detections:
            # With a letterboxed input the image only fills letterbox_size of
            # the model input (top left), so normalise against that.
            norm = getattr(self.model, "letterbox_size", None) or self.model.resize
            bbox[..., (0, 2)] /= norm[0]
            bbox[..., (1, 3)] /= norm[1]

        for b in bbox:
            if b[5] > self.model.viz_threshold:
                if type(self.model.label_offset) == dict:
                    class_name_idx = self.model.label_offset[int(b[4])]
                else:
                    class_name_idx = self.model.label_offset + int(b[4])

                if class_name_idx in self.model.dataset_info:
                    class_name = self.model.dataset_info[class_name_idx].name
                    if not class_name:
                        class_name = "UNDEFINED"
                    if self.model.dataset_info[class_name_idx].supercategory:
                        class_name = (
                            self.model.dataset_info[class_name_idx].supercategory
                            + "/"
                            + class_name
                        )
                    color = self.model.dataset_info[class_name_idx].rgb_color
                else:
                    class_name = "UNDEFINED"
                    color = (20, 220, 20)

                img = self.overlay_bounding_box(img, b, class_name, color)

        if self.debug:
            self.debug.log(self.debug_str)
            self.debug_str = ""

        return img

    def overlay_bounding_box(self, frame, box, class_name, color):
        """
        draw bounding box at given co-ordinates.

        Args:
            frame (numpy array): Input image where the overlay should be drawn
            bbox : Bounding box co-ordinates in format [X1 Y1 X2 Y2]
            class_name : Name of the class to overlay
        """
        box = [
            int(box[0] * frame.shape[1]),
            int(box[1] * frame.shape[0]),
            int(box[2] * frame.shape[1]),
            int(box[3] * frame.shape[0]),
        ]

        box_color = color
        luma = ((66*(color[0])+129*(color[1])+25*(color[2])+128)>>8)+16
        if(luma >= 128):
            text_color = (0, 0, 0)
        else:
            text_color = (255, 255, 255)

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), box_color, 2)
        cv2.rectangle(
            frame,
            (int((box[2] + box[0]) / 2) - 5, int((box[3] + box[1]) / 2) + 5),
            (int((box[2] + box[0]) / 2) + 160, int((box[3] + box[1]) / 2) - 15),
            box_color,
            -1,
        )
        cv2.putText(
            frame,
            class_name,
            (int((box[2] + box[0]) / 2), int((box[3] + box[1]) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
        )

        if self.debug:
            self.debug_str += class_name
            self.debug_str += str(box) + "\n"

        return frame


class PostProcessSegmentation(PostProcess):
    def __call__(self, img, results):
        """
        Post process function for segmentation
        Args:
            img: Input frame
            results: output of inference
        """
        img = self.blend_segmentation_mask(img, results[0])

        return img

    def blend_segmentation_mask(self, frame, results):
        """
        Process the result of the semantic segmentation model and return
        an image color blended with the mask representing different color
        for each class

        Args:
            frame (numpy array): Input image in BGR format which should be blended
            results (numpy array): Results of the model run
        """

        mask = np.squeeze(results)

        if len(mask.shape) > 2:
            mask = mask[0]

        if self.debug:
            self.debug_str += str(mask.flatten()) + "\n"
            self.debug.log(self.debug_str)
            self.debug_str = ""

        # Resize the mask to the original image for blending
        org_image_rgb = frame
        org_width = frame.shape[1]
        org_height = frame.shape[0]

        mask_image_rgb = self.gen_segment_mask(mask)
        mask_image_rgb = cv2.resize(
            mask_image_rgb, (org_width, org_height), interpolation=cv2.INTER_LINEAR
        )

        blend_image = cv2.addWeighted(
            mask_image_rgb, 1 - self.model.alpha, org_image_rgb, self.model.alpha, 0
        )

        return blend_image

    def gen_segment_mask(self, inp):
        """
        Generate the segmentation mask from the result of semantic segmentation
        model. Creates an RGB image with different colors for each class.

        Args:
            inp (numpy array): Result of the model run
        """

        r_map = (inp * 10).astype(np.uint8)
        g_map = (inp * 20).astype(np.uint8)
        b_map = (inp * 30).astype(np.uint8)

        return cv2.merge((r_map, g_map, b_map))

class PostProcessKeypointDetection(PostProcess):

    def __init__(self, flow):
        super().__init__(flow)

    def __call__(self, img, results):
        """
        Post process function for keypoint detection
        Args:
            img: Input frame
            results: output of inference
        """
        output = np.squeeze(results[0])

        scale_x = img.shape[1] / self.model.resize[0]
        scale_y = img.shape[0] / self.model.resize[1]

        det_bboxes, det_scores, det_labels, kpts = (
            np.array(output[:, 0:4]),
            np.array(output[:, 4]),
            np.array(output[:, 5]),
            np.array(output[:, 6:]),
        )
        for idx in range(len(det_bboxes)):
            det_bbox = det_bboxes[idx]
            kpt = kpts[idx]
            if det_scores[idx] > self.model.viz_threshold:
                det_bbox[..., (0, 2)] *= scale_x
                det_bbox[..., (1, 3)] *= scale_y

                # Drawing bounding box
                img = cv2.rectangle(
                    img,
                    (int(det_bbox[0]), int(det_bbox[1])),
                    (int(det_bbox[2]), int(det_bbox[3])),
                    (0, 255, 0),
                    2,
                )

                dataset_idx = int(det_labels[idx])
                # Put Label
                if type(self.model.label_offset) == dict:
                    dataset_idx = self.model.label_offset[dataset_idx]
                else:
                    dataset_idx = self.model.label_offset + dataset_idx

                if dataset_idx in self.model.dataset_info:
                    class_name = self.model.dataset_info[dataset_idx].name
                    if not class_name:
                        class_name = "UNDEFINED"
                    if self.model.dataset_info[dataset_idx].supercategory:
                        class_name = (
                            self.model.dataset_info[dataset_idx].supercategory
                            + "/"
                            + class_name
                        )
                    skeleton = self.model.dataset_info[dataset_idx].skeleton
                    if not skeleton:
                        skeleton = []

                else:
                    class_name = "UNDEFINED"
                    skeleton = []

                cv2.putText(
                    img,
                    class_name,
                    (int(det_bbox[0]), int(det_bbox[1]) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

                # Drawing keypoints
                num_kpts = len(kpt) // 3
                for kidx in range(num_kpts):
                    kx, ky, conf = kpt[3 * kidx], kpt[3 * kidx + 1], kpt[3 * kidx + 2]
                    kx = int(kx * scale_x)
                    ky = int(ky * scale_y)
                    if conf > 0.5:
                        cv2.circle(img, (kx, ky), 3, (255, 0, 0), -1)

                # Drawing connections between keypoints
                for sk in skeleton:
                    pos1 = (kpt[(sk[0] - 1) * 3], kpt[(sk[0] - 1) * 3 + 1])
                    pos1 = (int(pos1[0] * scale_x), int(pos1[1] * scale_y))

                    pos2 = (kpt[(sk[1] - 1) * 3], kpt[(sk[1] - 1) * 3 + 1])
                    pos2 = (int(pos2[0] * scale_x), int(pos2[1] * scale_y))

                    conf1 = kpt[(sk[0] - 1) * 3 + 2]
                    conf2 = kpt[(sk[1] - 1) * 3 + 2]
                    if conf1 > 0.5 and conf2 > 0.5:
                        cv2.line(img, pos1, pos2, (255, 0, 0), 1)


        return img


class VisualTrack:
    """
    Stand-in for a Norfair TrackedObject, produced by the hybrid mode.

    The hybrid mode follows its target with an OpenCV visual tracker instead
    of Norfair, but everything downstream of the tracker (coordinate output,
    motion trail, box drawing) only ever reads three attributes. Exposing the
    same three here keeps all of that code identical for both modes rather
    than sprinkling it with "if hybrid" branches.
    """

    __slots__ = ("estimate", "id", "age")

    def __init__(self, box, track_id, age):
        self.estimate = np.array(
            [[box[0], box[1]], [box[2], box[3]]], dtype=float
        )
        self.id = track_id
        self.age = age




class PanTiltServoController:
    """
    Pan/tilt servo loop for the T3 Gemstone.

    The camera sits on the servos, so dx/dy is the error left over after the
    last move. Every frame the commanded angle is nudged by a fixed fraction
    (gain) of that error, converted to degrees through the camera field of
    view. Two things shape the gain:

      - The picture reaches us several frames after it was taken, so the error
        we see still contains moves we already made. A gain of 1.0 (correct the
        whole error at once) therefore overshoots and oscillates. 0.08-0.15 is
        the stable range at 30 fps.
      - max_step caps a single update so a jumping box cannot yank the rig.

    The commanded angle accumulates continuously. A pulse is only rewritten
    when its width actually changes, so a target at rest does not keep
    restarting the PWM.

    Two output backends:

      hwpwm  Hardware PWM through /sys/class/pwm (the default). The pulses
             come from the epwm peripheral, so a busy CPU cannot bend them.
             Needs the servo signals on PWM capable pins and the matching
             overlay, e.g. pwm-epwm1-gpio6-gpio13.dtbo.
      lgpio  Software PWM on any GPIO. lgpio times the pulses from its own
             thread; a pulse 100 us late is a ~9 degree twitch, and this
             process keeps the CPU busy, so that thread is started with
             real-time priority (see _init_lgpio). Still jitters under load.
    """

    def __init__(
        self,
        backend="hwpwm",
        pwm_device="23010000.pwm",  # epwm1
        pan_pwm_channel=1,   # GPIO13, header pin 33
        tilt_pwm_channel=0,  # GPIO6, header pin 31
        pan_chip=3,          # lgpio: GPIO17
        pan_line=8,
        tilt_chip=2,         # lgpio: GPIO27
        tilt_line=33,
        deadzone=0.03,       # Error (normalized) that counts as centred
        gain=0.10,           # Fraction of the error corrected per frame
        max_step=1.5,        # Largest single move, degrees per frame
        smoothing=0.3,       # EMA on the error (0.0 = raw, 1.0 = frozen)
        hfov=40.0,           # Horizontal camera field of view, degrees
        vfov=23.0,           # Vertical camera field of view, degrees
        release_after=0.0,   # Cut pulses after this many seconds at rest, 0 = never
        rt_priority=70,      # SCHED_FIFO priority of the pulse thread, 0 = off
        pan_range=(-90.0, 90.0),
        tilt_range=(-50.0, 50.0),
        pan_invert=False,
        tilt_invert=False,
        search=True,         # Sweep the pan axis while no target is seen
        search_delay=1.0,    # Seconds without a target before sweeping
        search_step=10.0,    # Degrees per hop of the stop-and-go sweep
        search_dwell=0.8,    # Seconds to stand still after each hop
        search_pan=(-60.0, 60.0),  # Sweep limits, degrees
        search_tilt=0.0,     # Tilt angle held while sweeping
    ):
        self.backend = backend
        self.pwm_device = pwm_device
        self.pan_pwm_channel = pan_pwm_channel
        self.tilt_pwm_channel = tilt_pwm_channel
        self.pan_chip = pan_chip
        self.pan_line = pan_line
        self.tilt_chip = tilt_chip
        self.tilt_line = tilt_line
        self.deadzone = deadzone
        self.gain = gain
        self.max_step = max_step
        self.smoothing = smoothing
        self.half_hfov = hfov / 2.0
        self.half_vfov = vfov / 2.0
        self.release_after = release_after
        self.rt_priority = rt_priority
        self.pan_range = pan_range
        self.tilt_range = tilt_range
        self.pan_invert = pan_invert
        self.tilt_invert = tilt_invert
        self.search_enabled = search
        self.search_delay = search_delay
        self.search_step = search_step
        self.search_dwell = search_dwell
        self.search_pan = (min(search_pan), max(search_pan))
        self.search_tilt = search_tilt

        # Sweep state. search_dir is +1/-1 in pan angle, and is pointed by
        # update() towards the side the target was last seen on, so a target
        # that walks out of the frame is looked for where it went first.
        self.searching = False
        self.search_dir = 1.0
        self.last_target_time = 0.0
        self.next_hop_time = None

        self.pan_angle = 0.0
        self.tilt_angle = 0.0
        # Last pulse width written to each line. idle() silences the lines
        # but keeps these, so a line only wakes up when its angle changes.
        self.pan_pw = None
        self.tilt_pw = None
        self.pulsing = False

        # Filtrelenmiş hata
        self.smooth_dx = 0.0
        self.smooth_dy = 0.0

        self.last_move_time = 0.0

        self.pan_out = None
        self.tilt_out = None
        self.enabled = False

        self._init_servos()

    def _angle_to_pw(self, angle):
        """Map angle in degrees to pulse width in microseconds (500 to 2500 us)."""
        pw = 1500.0 + (angle / 90.0) * 1000.0
        return max(500, min(2500, int(round(pw))))

    def _init_servos(self):
        import time

        try:
            if self.backend == "hwpwm":
                self.pan_out = _SysfsPwmServo(self.pwm_device, self.pan_pwm_channel)
                self.tilt_out = _SysfsPwmServo(self.pwm_device, self.tilt_pwm_channel)
                rt = None
            elif self.backend == "lgpio":
                rt = self._init_lgpio()
            else:
                raise ValueError("unknown servo_backend '%s'" % self.backend)

            # Başlangıçta 0'a al
            self.enabled = True
            self._send(0.0, 0.0)
            time.sleep(0.4)
            self.last_move_time = time.time()
            print(
                "[INFO] Pan-Tilt Servos initialized (%s, gain=%.2f "
                "deadzone=%.2f fov=%.0fx%.0f%s)."
                % (self.backend, self.gain, self.deadzone, self.half_hfov * 2,
                   self.half_vfov * 2,
                   "" if rt is None else " rt=" + ("on" if rt else "off"))
            )
        except Exception as exc:
            self.enabled = False
            print(
                "[ERROR] Pan-Tilt Servo initialization failed: %s" % exc
            )

    def _init_lgpio(self):
        """Open the lgpio lines. Returns whether the pulse thread got RT."""
        import os

        # lgpio starts its pulse thread from whichever thread first opens a
        # chip or sends a pulse, and the new thread inherits that thread's
        # scheduling policy. Switch to SCHED_FIFO for exactly that window and
        # back to normal afterwards, so only the pulse thread ends up
        # real-time and the rest of the app (numpy, GStreamer) stays as it was.
        rt = False
        if self.rt_priority > 0:
            try:
                os.sched_setscheduler(
                    0, os.SCHED_FIFO, os.sched_param(self.rt_priority)
                )
                rt = True
            except (AttributeError, OSError) as exc:
                print("[WARNING] Servo pulse thread stays non-RT: %s" % exc)

        try:
            import lgpio

            pan_handle = lgpio.gpiochip_open(self.pan_chip)
            tilt_handle = (
                pan_handle
                if self.tilt_chip == self.pan_chip
                else lgpio.gpiochip_open(self.tilt_chip)
            )
            self.pan_out = _LgpioServo(lgpio, pan_handle, self.pan_line)
            self.tilt_out = _LgpioServo(lgpio, tilt_handle, self.tilt_line)
            # The first pulse is what starts the thread, keep it in the window.
            self.pan_out.write(1500)
            self.tilt_out.write(1500)
        finally:
            if rt:
                os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
        return rt

    def _send(self, pan_angle, tilt_angle):
        """Write both angles, touching a line only if its pulse changed."""
        moved = False
        pan_pw = self._angle_to_pw(pan_angle)
        if pan_pw != self.pan_pw:
            try:
                self.pan_out.write(pan_pw)
                self.pan_pw = pan_pw
                self.pulsing = moved = True
            except Exception:
                pass
        tilt_pw = self._angle_to_pw(tilt_angle)
        if tilt_pw != self.tilt_pw:
            try:
                self.tilt_out.write(tilt_pw)
                self.tilt_pw = tilt_pw
                self.pulsing = moved = True
            except Exception:
                pass
        return moved

    def _step(self, error, half_fov):
        """Degrees to move for a normalized error, 0 inside the deadzone."""
        if abs(error) <= self.deadzone:
            return 0.0
        step = self.gain * error * half_fov
        return max(-self.max_step, min(self.max_step, step))

    def update(self, raw_dx, raw_dy):
        """One control step towards the target at (raw_dx, raw_dy)."""
        import time

        if not self.enabled:
            return

        now = time.time()
        self.last_target_time = now
        self.searching = False
        self.next_hop_time = None
        if abs(raw_dx) > self.deadzone:
            # The pan angle moves this way to follow a target on this side,
            # see the pan step below.
            toward = -1.0 if raw_dx > 0 else 1.0
            self.search_dir = -toward if self.pan_invert else toward

        # Düşük geçiren filtre (kutu titreşimini süzer). Kept light: every bit
        # of smoothing is extra delay inside the loop.
        self.smooth_dx = (
            self.smoothing * self.smooth_dx + (1.0 - self.smoothing) * raw_dx
        )
        self.smooth_dy = (
            self.smoothing * self.smooth_dy + (1.0 - self.smoothing) * raw_dy
        )

        # 1. PAN (Yatay Takip). Target right of centre (dx > 0) lowers the
        # angle unless inverted.
        delta_pan = self._step(self.smooth_dx, self.half_hfov)
        if self.pan_invert:
            delta_pan = -delta_pan
        self.pan_angle = max(
            self.pan_range[0], min(self.pan_range[1], self.pan_angle - delta_pan)
        )

        # 2. TILT (Dikey Takip). Target below centre (dy > 0) lowers the
        # angle unless inverted.
        delta_tilt = self._step(self.smooth_dy, self.half_vfov)
        if self.tilt_invert:
            delta_tilt = -delta_tilt
        self.tilt_angle = max(
            self.tilt_range[0],
            min(self.tilt_range[1], self.tilt_angle - delta_tilt),
        )

        if self._send(self.pan_angle, self.tilt_angle):
            self.last_move_time = now
        elif 0 < self.release_after < now - self.last_move_time:
            # Hedef merkezlendi: Sessiz bekleme
            self.idle()

    def search(self):
        """
        No target: hold for search_delay, then sweep the pan axis between the
        search_pan limits in hops, while tilt goes to search_tilt.

        Stop and go rather than a continuous sweep: indoors the exposure is
        long, and at 20 deg/s every frame of a moving camera was smeared
        enough that the face model found nothing even with a face filling
        the frame. Each hop takes the servo ~0.1 s; the rig then stands still
        for search_dwell, which at 15 fps is ~12 sharp frames, more than
        initialization_delay needs to confirm a detection.
        """
        import time

        if not self.enabled:
            return
        now = time.time()
        if not self.search_enabled or now - self.last_target_time < self.search_delay:
            self.hold()
            return

        self.searching = True
        if self.next_hop_time is not None and now < self.next_hop_time:
            return
        self.next_hop_time = now + self.search_dwell

        lo, hi = self.search_pan
        pan = self.pan_angle + self.search_dir * self.search_step
        if pan >= hi:
            pan, self.search_dir = hi, -1.0
        elif pan <= lo:
            pan, self.search_dir = lo, 1.0
        self.pan_angle = max(self.pan_range[0], min(self.pan_range[1], pan))
        self.tilt_angle = self.search_tilt

        if self._send(self.pan_angle, self.tilt_angle):
            self.last_move_time = now

    # Fire recoil: (right, up) degrees applied on consecutive frames. Kick up
    # and shake sideways, then settle back; each column sums to zero, so the
    # camera ends where it started.
    RECOIL = ((1.5, 5.0), (-3.0, -1.5), (2.0, -1.5), (-0.5, -1.2), (0.0, -0.8))

    def recoil(self):
        self.recoil_steps = list(self.RECOIL)

    def step_recoil(self):
        """Apply the next recoil step; True while a recoil is running."""
        steps = getattr(self, "recoil_steps", None)
        if not steps:
            return False
        right, up = steps.pop(0)
        self.nudge(right, up)
        return True

    def nudge(self, right_deg, up_deg):
        """
        Manual move: turn the view right/up by the given degrees (negative
        for left/down), with the same signs update() uses to follow a target
        on that side, so the invert flags apply here too.
        """
        import time

        if not self.enabled:
            return
        d_pan = right_deg * (-1.0 if self.pan_invert else 1.0)
        self.pan_angle = max(
            self.pan_range[0], min(self.pan_range[1], self.pan_angle - d_pan)
        )
        d_tilt = -up_deg * (-1.0 if self.tilt_invert else 1.0)
        self.tilt_angle = max(
            self.tilt_range[0], min(self.tilt_range[1], self.tilt_angle - d_tilt)
        )
        if self._send(self.pan_angle, self.tilt_angle):
            self.last_move_time = time.time()

    def hold(self):
        """No target: keep the current angle, go silent once at rest."""
        import time

        if self.enabled and 0 < self.release_after < time.time() - self.last_move_time:
            self.idle()

    def idle(self):
        """Silence pulses to stop buzzing. Keeps current angle."""
        if not self.enabled or not self.pulsing:
            return
        try:
            self.pan_out.off()
            self.tilt_out.off()
            self.pulsing = False
        except Exception:
            pass

    def reset_filters(self):
        """Reset smoothing filters when acquiring a new target."""
        self.smooth_dx = 0.0
        self.smooth_dy = 0.0

    def close(self):
        if not self.enabled:
            return
        for out in (self.pan_out, self.tilt_out):
            try:
                out.close()
            except Exception:
                pass


class _SysfsPwmServo:
    """One servo on a hardware PWM channel, via /sys/class/pwm."""

    PERIOD_NS = 20000000  # 50 Hz

    def __init__(self, device, channel):
        import glob
        import os
        import time

        # pwmchipN numbers follow probe order, so find the chip by the
        # address of the peripheral instead.
        chips = [
            c for c in glob.glob("/sys/class/pwm/pwmchip*")
            if os.path.basename(os.path.realpath(c + "/device")) == device
        ]
        if not chips:
            raise RuntimeError("no pwmchip for %s" % device)
        self.path = "%s/pwm%d" % (chips[0], channel)
        if not os.path.isdir(self.path):
            self._put(chips[0] + "/export", channel)
            # udev may still be fixing up the new directory.
            for _ in range(50):
                if os.access(self.path + "/period", os.W_OK):
                    break
                time.sleep(0.01)
        self._put("enable", 0)
        # duty_cycle may not exceed period, so clear it before the period.
        self._put("duty_cycle", 0)
        self._put("period", self.PERIOD_NS)
        self.enabled = False

    def _put(self, name, value):
        path = name if name.startswith("/") else "%s/%s" % (self.path, name)
        with open(path, "w") as f:
            f.write(str(value))

    def write(self, pw_us):
        self._put("duty_cycle", int(pw_us) * 1000)
        if not self.enabled:
            self._put("enable", 1)
            self.enabled = True

    def off(self):
        self._put("enable", 0)
        self.enabled = False

    def close(self):
        self.off()


class _LgpioServo:
    """One servo on any GPIO, software PWM through lgpio."""

    def __init__(self, lgpio, handle, line):
        self.lgpio = lgpio
        self.handle = handle
        self.line = line
        try:
            lgpio.gpio_claim_output(handle, line)
        except Exception:
            pass

    def write(self, pw_us):
        self.lgpio.tx_servo(self.handle, self.line, int(pw_us), 50)

    def off(self):
        self.lgpio.tx_servo(self.handle, self.line, 0)

    def close(self):
        self.off()
        try:
            self.lgpio.gpiochip_close(self.handle)
        except Exception:
            pass


class PostProcessTracking(PostProcess):
    """
    Post process for detection models that adds Norfair based multi object
    tracking on top of the detections.

    Selected for a detection model when the config sets "enable_tracking: True".
    The detection decode stage of __call__ is deliberately identical to
    PostProcessDetection so both stay in sync with the model output API.

    Everything is drawn in place, so the returned frame has exactly the same
    size as the frame handed in and a tracking subflow is sized like any other.
    """

    # Norfair tracker tuning. hit_counter_max is how many frames an id
    # survives without a matching detection before it is dropped. It used to
    # be 5 (~0.17 s at 30 fps), which was short enough that a stationary
    # person lost and regained identity several times a minute; every one of
    # those renames breaks anything downstream that locked onto an id. One
    # second of grace costs nothing and removes most of the churn.
    DISTANCE_THRESHOLD_BBOX = 1.0
    DISTANCE_FUNCTION = "iou"
    INITIALIZATION_DELAY = 4
    HIT_COUNTER_MAX = 30

    # Number of past positions kept per object for the motion trail
    PATH_HISTORY_SIZE = 10

    # Dataset class names to track. Resolved to model class indices through
    # dataset_info in get_track_class_indices(). Overridden by the config key
    # "track_classes"; an empty list there means track every class.
    TRACK_CLASS_NAMES = ("person",)

    # ------------------------------------------------------------------
    # Tracking modes, selected with the config key "track_mode".
    #
    # multi   Every object of the tracked classes gets its own id, box and
    #         coordinate line. The original behaviour, and the default.
    #
    # lock    Detection keeps running on every frame, but the flow commits to
    #         ONE target and reports only that one. When the target's Norfair
    #         id dies the lock is re-acquired from the detection nearest the
    #         last known position, so an id rename no longer loses the target.
    #         Costs nothing extra: detection runs on the C7x either way.
    #
    # hybrid  The target is detected once, then handed to an OpenCV visual
    #         tracker that follows the image patch without any help from the
    #         detector. Detection is re-run every redetect_interval frames to
    #         correct the drift a visual tracker always accumulates. This is
    #         the only mode that can follow something the model cannot name,
    #         but it runs on the A53 CPU and can cost frames per second.
    # ------------------------------------------------------------------
    MODE_MULTI = "multi"
    MODE_LOCK = "lock"
    MODE_HYBRID = "hybrid"
    TRACK_MODE = MODE_MULTI

    # Which candidate becomes the target when there is no lock yet.
    #   closest_to_center  the one a pan/tilt rig has to move least to reach
    #   largest            the biggest box, usually the nearest object
    #   first              whatever the tracker lists first, cheapest
    LOCK_POLICY = "closest_to_center"

    # How far, as a fraction of the frame diagonal, a candidate may be from
    # the last known target position and still be accepted as the same target
    # when re-acquiring. Large enough to survive a rename, small enough not to
    # jump to a different object across the room.
    LOCK_REACQUIRE_RADIUS = 0.25

    # How many frames after a lock is lost the last target position still
    # decides who gets the next lock. A fast pan blurs the target for a few
    # frames, its id dies and the new one needs initialization_delay frames;
    # without this memory the next lock went to whoever was closest to the
    # frame centre, which with two people in view is not necessarily the one
    # just followed.
    LOCK_MEMORY_FRAMES = 30

    # hybrid: frames between detector corrections, and the IoU below which a
    # correction is treated as "the visual tracker is on the wrong thing".
    REDETECT_INTERVAL = 30
    REDETECT_MIN_IOU = 0.2

    # hybrid: which OpenCV tracker, and at what scale to run it.
    #
    # Measured on this board, 720p frame with a 160x320 box, ms per frame:
    #
    #                 full res   640x360   320x180
    #     CSRT          375        188       166
    #     KCF           217         50        49
    #     MOSSE          29        7.3       1.9
    #
    # CSRT and KCF are both far too slow here: the pipeline only has ~35 ms
    # per frame in total, so either of them would drop 30 fps to single
    # digits. Note CSRT barely improves when the frame shrinks, because its
    # cost is the filter it learns rather than the area it scans.
    #
    # MOSSE is the least robust of the three on its own, but this mode re-runs
    # the detector every REDETECT_INTERVAL frames precisely to correct the
    # drift, so robustness comes from the correction rather than the tracker.
    #
    # Tracking on a downscaled copy loses nothing that matters here: what the
    # controller consumes is dx/dy, and a few pixels of box edge do not move
    # it meaningfully.
    #
    # 0.5 is the measured sweet spot for the whole hybrid step, resize
    # included:
    #
    #     scale 0.50   9.5 ms/frame   worst drift 10 px between corrections
    #     scale 0.25  17.0 ms/frame   worst drift 22 px
    #
    # Scaling down further is slower, not faster, which is worth knowing
    # before someone "optimises" it: halving again turns each output pixel
    # into an average of 16 source pixels instead of 4, and that costs more in
    # the resize than it saves in the tracker.
    TRACKER_ALGO = "mosse"
    TRACKER_SCALE = 0.5

    # Status badge drawn top left: which mode is running and what it is doing
    # right now. Costs one small rectangle and one line of text per frame.
    SHOW_STATUS_BADGE = True

    # How often the active objects are printed, in frames. Appearing and
    # disappearing ids are printed as they happen regardless. Set to 0 to
    # silence the terminal completely; tracked_positions is still updated
    # every frame, so a consumer reading the attribute is unaffected.
    PRINT_EVERY_N_FRAMES = 15

    def __init__(self, flow):
        super().__init__(flow)

        # Imported here rather than at module scope; see the note at the top of
        # this file.
        from norfair import Detection, Tracker
        from norfair.drawing.color import Palette
        from path_draw import PathDraw

        self.Detection = Detection
        # Same palette PathDraw uses, so an object's box, label and motion
        # trail all share one stable colour for as long as the id lives.
        self.Palette = Palette

        self.tracker = Tracker(
            initialization_delay=int(
                getattr(self.model, "initialization_delay",
                        self.INITIALIZATION_DELAY)
            ),
            distance_function=self.DISTANCE_FUNCTION,
            distance_threshold=self.DISTANCE_THRESHOLD_BBOX,
            hit_counter_max=int(
                getattr(self.model, "hit_counter_max", self.HIT_COUNTER_MAX)
            ),
        )

        self.pathd = PathDraw(history_size=self.PATH_HISTORY_SIZE)

        self.track_class_indices = self.get_track_class_indices()

        # ------------------------------------------------------------------
        # Mode configuration and state
        # ------------------------------------------------------------------
        m = self.model
        mode = str(getattr(m, "track_mode", self.TRACK_MODE)).lower()
        if mode not in (self.MODE_MULTI, self.MODE_LOCK, self.MODE_HYBRID):
            print(
                "[WARNING] Unknown track_mode '%s', falling back to '%s'. "
                "Valid values: %s, %s, %s."
                % (mode, self.MODE_MULTI, self.MODE_MULTI, self.MODE_LOCK,
                   self.MODE_HYBRID)
            )
            mode = self.MODE_MULTI
        self.track_mode = mode
        self.lock_policy = str(getattr(m, "lock_policy", self.LOCK_POLICY)).lower()
        self.redetect_interval = int(
            getattr(m, "redetect_interval", self.REDETECT_INTERVAL)
        )

        # Target lock state, shared by lock and hybrid.
        self.locked_id = None
        # Last known target box in frame pixels, [x1, y1, x2, y2]. Kept even
        # while the lock is lost so re-acquisition has something to aim at.
        self.locked_box = None
        # Frame the lock was last lost on, for LOCK_MEMORY_FRAMES.
        self.lock_lost_frame = None
        # Human readable state, drawn in the badge and used by report().
        self.state = "SEARCHING"

        # hybrid only: the OpenCV visual tracker instance, created the moment
        # a target is picked and destroyed when it fails.
        self.cv_tracker = None
        self.frames_since_redetect = 0
        self.hybrid_age = 0
        self.tracker_scale = float(
            getattr(self.model, "tracker_scale", self.TRACKER_SCALE)
        )
        if not 0.05 <= self.tracker_scale <= 1.0:
            print(
                "[WARNING] tracker_scale %.3f out of range, using %.2f."
                % (self.tracker_scale, self.TRACKER_SCALE)
            )
            self.tracker_scale = self.TRACKER_SCALE

        if self.track_mode == self.MODE_HYBRID:
            (
                self._cv_tracker_factory,
                self._cv_tracker_name,
            ) = self.get_cv_tracker_factory()

        # The badge shows only the state, so say once here which mode produced
        # it. Also the single line to look at when a run does not behave the
        # way the config was meant to.
        described = {
            self.MODE_MULTI: "multi - a separate id for every object, no lock",
            self.MODE_LOCK: "lock - locks onto a single target",
            self.MODE_HYBRID: "hybrid - visual tracking with periodic correction",
        }
        print("[INFO] Tracking mode: %s" % described[self.track_mode])

        # Position of every tracked object in the current frame, refreshed by
        # update_tracked_positions() once per frame. See that method for the
        # meaning of dx/dy. Read it from another thread to drive something
        # else (a servo, a socket, a log) off the tracker.
        self.tracked_positions = []

        self.frame_count = 0
        self._reported_ids = set()

        # Local panel through the framebuffer, see fb_view.py. "" turns it off.
        self.fb_view = None
        fb_dev = str(getattr(self.model, "display_fb", ""))
        if fb_dev:
            try:
                from fb_view import FbView

                self.fb_view = FbView(
                    device=fb_dev, rotate=int(getattr(self.model, "display_rotate", 90))
                )
            except Exception as exc:
                print("[WARNING] Panel view disabled: %s" % exc)

        # Touch on the panel: tap a tracked object to lock onto it, tap
        # anywhere else to drop the lock and let lock_policy pick again. The
        # tap arrives on the touch thread and is only recorded here; the next
        # frame acts on it in update_lock(). "" turns it off.
        self.touch_request = None
        self.touch_marker = None
        self.touch = None
        self.ui = None
        self.manual_speed = float(getattr(self.model, "servo_manual_speed", 30.0))
        # Picture style, see hud.py. The panel's effect buttons override it.
        from hud import Hud, STYLES

        self.hud = Hud()
        self.hud_style = str(getattr(self.model, "hud_style", "aim")).lower()
        if self.hud_style not in STYLES:
            print("[WARNING] Unknown hud_style '%s', using aim." % self.hud_style)
            self.hud_style = "aim"
        self._manual_time = None
        self._all_tracked = []
        # Fire button state: frame of the last shot, its result, the score.
        self.fire_frame = None
        self.fire_hit = False
        self.fire_point = None
        self.score = 0
        touch_dev = str(getattr(self.model, "display_touch", "auto"))
        if self.fb_view and touch_dev:
            try:
                from touch_input import TouchInput

                from panel_ui import PanelUI

                def on_picture_tap(px, py):
                    pt = self.fb_view.panel_to_frame(px, py)
                    if pt is not None:
                        self.touch_request = pt
                        self.touch_marker = (pt, self.frame_count)

                self.ui = PanelUI(
                    self.fb_view.width, self.fb_view.height, on_picture_tap,
                    effect=self.hud_style,
                )
                self.fb_view.ui = self.ui
                self.fb_view.filter = lambda f: self.hud.filter(f, self.ui.effect)
                self.touch = TouchInput(
                    self.ui.down, self.ui.up, device=touch_dev
                )
            except Exception as exc:
                print("[WARNING] Touch input disabled: %s" % exc)

        # Browser view of the drawn frames, see web_view.py. 0 turns it off.
        self.web_view = None
        web_port = int(getattr(self.model, "web_view_port", 8080))
        if web_port:
            try:
                from web_view import WebView

                self.web_view = WebView(port=web_port)
            except Exception as exc:
                print("[WARNING] Web view disabled: %s" % exc)

        # ------------------------------------------------------------------
        # Pan-Tilt Servo Controller (hwpwm: GPIO13 pan, GPIO6 tilt)
        # ------------------------------------------------------------------
        pan_chip = int(getattr(m, "servo_pan_chip", 3))
        pan_line = int(getattr(m, "servo_pan_line", 8))
        tilt_chip = int(getattr(m, "servo_tilt_chip", 2))
        tilt_line = int(getattr(m, "servo_tilt_line", 33))
        pan_invert = bool(getattr(m, "servo_pan_invert", False))
        tilt_invert = bool(getattr(m, "servo_tilt_invert", False))
        # Where inside the box to aim, as a fraction of its height from the
        # top. 0.5 is the centre; for a person something like 0.15 puts the
        # head, not the chest, in the middle of the frame. A person close to
        # the camera fills the whole frame height, so the box centre sits on
        # the frame centre and would never move the tilt axis at all.
        self.servo_aim_y = float(getattr(m, "servo_aim_y", 0.5))
        # One line per frame while locked: detection score, live flag, error
        # and the commanded angles. For tuning; off by default.
        self.servo_debug = bool(getattr(m, "servo_debug", False))

        # Camera motion compensation. The camera rides on the servos, so every
        # pan step shifts the whole picture (~32 px per degree), and Norfair's
        # Kalman filter learned that shift as the target's own velocity. When
        # a detection then dropped out the prediction flew off in the
        # direction of the last pan move, the next detection no longer matched
        # it, and the lock was lost after one or two seconds every time.
        # Tracking in "world" coordinates (image position plus where the
        # servos point) removes the shift. The picture lags the command, so
        # the angle from servo_motion_delay frames ago is the one that
        # belongs to the current frame.
        from collections import deque
        from norfair.camera_motion import TranslationTransformation

        self.TranslationTransformation = TranslationTransformation
        self.motion_delay = max(0, int(getattr(m, "servo_motion_delay", 2)))
        self.angle_history = deque(maxlen=self.motion_delay + 1)
        self.motion_compensation = bool(getattr(m, "servo_motion_compensation", True))

        self.servo_controller = PanTiltServoController(
            backend=str(getattr(m, "servo_backend", "hwpwm")).lower(),
            pwm_device=str(getattr(m, "servo_pwm_device", "23010000.pwm")),
            pan_pwm_channel=int(getattr(m, "servo_pan_pwm_channel", 1)),
            tilt_pwm_channel=int(getattr(m, "servo_tilt_pwm_channel", 0)),
            pan_chip=pan_chip,
            pan_line=pan_line,
            tilt_chip=tilt_chip,
            tilt_line=tilt_line,
            deadzone=float(getattr(m, "servo_deadzone", 0.03)),
            gain=float(getattr(m, "servo_gain", 0.10)),
            max_step=float(getattr(m, "servo_max_step", 1.5)),
            smoothing=float(getattr(m, "servo_smoothing", 0.3)),
            hfov=float(getattr(m, "servo_hfov", 40.0)),
            vfov=float(getattr(m, "servo_vfov", 23.0)),
            release_after=float(getattr(m, "servo_release_after", 0.0)),
            rt_priority=int(getattr(m, "servo_rt_priority", 70)),
            pan_invert=pan_invert,
            tilt_invert=tilt_invert,
            search=bool(getattr(m, "servo_search", True)),
            search_delay=float(getattr(m, "servo_search_delay", 1.0)),
            search_step=float(getattr(m, "servo_search_step", 10.0)),
            search_dwell=float(getattr(m, "servo_search_dwell", 0.8)),
            search_pan=tuple(
                float(a) for a in getattr(m, "servo_search_pan", (-60.0, 60.0))
            ),
            search_tilt=float(getattr(m, "servo_search_tilt", 0.0)),
        )

    def get_track_class_names(self):
        """
        Class names to track, from the config key "track_classes" if present.

        A single string is accepted as well as a list, so both
        "track_classes: person" and "track_classes: [person, dog]" work. An
        empty list is meaningful and means "track every class the model
        emits"; that is different from the key being absent, which keeps the
        TRACK_CLASS_NAMES default.
        """
        names = getattr(self.model, "track_classes", None)
        if names is None:
            return tuple(self.TRACK_CLASS_NAMES)
        if isinstance(names, str):
            names = [names]
        return tuple(str(n).strip() for n in names)

    def get_track_class_indices(self):
        """
        Resolve the tracked class names to the set of raw class indices the
        model emits. Returns None when every class is to be tracked.

        The raw index is mapped through label_offset into dataset_info and
        matched by name, which keeps the filter correct for any detection model
        rather than assuming a fixed index. Falls back to index 0 (person in
        the COCO trained models shipped with the SDK) if nothing matches.
        """
        wanted = self.get_track_class_names()
        if not wanted:
            print("[INFO] Tracking every class the model emits.")
            return None

        indices = set()
        dataset_info = getattr(self.model, "dataset_info", None)
        available = set()

        if dataset_info:
            if type(self.model.label_offset) == dict:
                # label_offset maps raw model index -> dataset index
                raw_to_dataset = list(self.model.label_offset.items())
            else:
                raw_to_dataset = [
                    (idx - self.model.label_offset, idx) for idx in dataset_info
                ]

            for raw_idx, dataset_idx in raw_to_dataset:
                if dataset_idx in dataset_info:
                    name = dataset_info[dataset_idx].name
                    available.add(name)
                    if name in wanted:
                        indices.add(int(raw_idx))

        # Name the ones that did not resolve rather than silently tracking
        # something else: a typo in track_classes is otherwise invisible until
        # someone wonders why nothing is ever tracked.
        missing = [n for n in wanted if n not in available]
        if missing and available:
            print(
                "[WARNING] These track_classes are not in the model's dataset: "
                "%s. Available names include: %s ..."
                % (", ".join(missing), ", ".join(sorted(available)[:12]))
            )

        if not indices:
            print(
                "[WARNING] Could not resolve %s in the model's dataset info. "
                "Falling back to class index 0 for tracking."
                % str(list(wanted))
            )
            indices.add(0)
        else:
            print(
                "[INFO] Tracking classes: %s (model indices %s)"
                % (", ".join(n for n in wanted if n in available),
                   sorted(indices))
            )

        return indices

    def __call__(self, img, results):
        """
        Post process function for people tracking
        Args:
            img: Input frame
            results: output of inference
        """
        # ------------------------------------------------------------------
        # Detection decode. Identical to PostProcessDetection.__call__().
        # ------------------------------------------------------------------
        for i, r in enumerate(results):
            r = np.squeeze(r)
            if r.ndim == 1:
                r = np.expand_dims(r, 1)
            results[i] = r

        if self.model.shuffle_indices:
            results_reordered = []
            for i in self.model.shuffle_indices:
                results_reordered.append(results[i])
            results = results_reordered

        if results[-1].ndim < 2:
            results = results[:-1]

        bbox = np.concatenate(results, axis=-1)

        if self.model.formatter:
            if self.model.ignore_index == None:
                bbox_copy = copy.deepcopy(bbox)
            else:
                bbox_copy = copy.deepcopy(np.delete(bbox, self.model.ignore_index, 1))
            bbox[..., self.model.formatter["dst_indices"]] = bbox_copy[
                ..., self.model.formatter["src_indices"]
            ]

        if not self.model.normalized_detections:
            # With a letterboxed input the image only fills letterbox_size of
            # the model input (top left), so normalise against that.
            norm = getattr(self.model, "letterbox_size", None) or self.model.resize
            bbox[..., (0, 2)] /= norm[0]
            bbox[..., (1, 3)] /= norm[1]

        # ------------------------------------------------------------------
        # OBJECT TRACKING
        # ------------------------------------------------------------------
        # PostProcessDetection scales normalized boxes to pixels inside
        # overlay_bounding_box(). The tracker, heat map and time counter all
        # work in absolute frame pixels, so scale up once here instead.
        bbox[..., (0, 2)] *= img.shape[1]
        bbox[..., (1, 3)] *= img.shape[0]

        self.frame_count += 1

        # Detections that exist in THIS frame, by identity. A Norfair object
        # whose last_detection is not among them is coasting on its Kalman
        # prediction; see update_pan_tilt(). None means every object is live
        # (hybrid, whose visual tracker measures the target every frame).
        self._live_detections = None

        if self.track_mode == self.MODE_HYBRID:
            # The detector is only consulted on correction frames, so bbox is
            # decoded above but usually thrown away. That waste is deliberate:
            # the decode is a handful of numpy operations on an array the
            # inference already produced, and skipping it would mean skipping
            # inference itself, which happens further up the pipeline and is
            # not ours to switch off.
            tracked_objects = self.update_hybrid(img, bbox)
        else:
            detections = self.yolo_detections_to_norfair_detections(bbox)
            self._live_detections = set(id(d) for d in detections)
            tracked_objects = self.tracker.update(
                detections=detections,
                coord_transformations=self.camera_motion(img.shape[1], img.shape[0]),
            )
            if self.track_mode == self.MODE_LOCK:
                self._all_tracked = tracked_objects
                tracked_objects = self.update_lock(tracked_objects, img)
            else:
                self.state = (
                    "TRACKING: %d objects" % len(tracked_objects)
                    if tracked_objects
                    else "SEARCHING"
                )

        self._last_tracked = tracked_objects
        self.update_tracked_positions(tracked_objects, img.shape[1], img.shape[0])
        self.report_tracked_positions()
        self.update_pan_tilt(img.shape[1], img.shape[0])
        if self.servo_controller and self.servo_controller.enabled:
            self.angle_history.append(
                (self.servo_controller.pan_angle, self.servo_controller.tilt_angle)
            )

        style = self.ui.effect if self.ui is not None else self.hud_style

        # Fire: a hit when the locked target's centre is near the crosshair.
        if self.ui is not None and self.ui.fire_request:
            self.ui.fire_request = False
            if self.fire_frame is None or self.frame_count - self.fire_frame >= 8:
                self.fire_frame = self.frame_count
                self.fire_hit, self.fire_point = False, None
                if self.state.startswith("LOCKED") and tracked_objects:
                    x1, y1, x2, y2 = self.box_of(tracked_objects[0])
                    tx, ty = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    self.fire_point = (int(tx), int(ty))
                    half = max(abs(x2 - x1), abs(y2 - y1)) / 2.0
                    dist = ((tx - img.shape[1] / 2.0) ** 2 + (ty - img.shape[0] / 2.0) ** 2) ** 0.5
                    self.fire_hit = dist <= max(60.0, half)
                if self.fire_hit:
                    self.score += 1
                if self.servo_controller and self.servo_controller.enabled:
                    self.servo_controller.recoil()
                print("[f:%d] ! fire: %s (score %d)" % (
                    self.frame_count, "hit" if self.fire_hit else "miss", self.score), flush=True)
        locked_ids = set(o.id for o in tracked_objects)
        others = [
            self.box_of(o) for o in self._all_tracked if o.id not in locked_ids
        ] if self.track_mode == self.MODE_LOCK else []

        if style == "plain":
            img = self.pathd.draw(img, tracked_objects)
            # The other candidates, thin and grey, so there is something to
            # tap on the panel to move the lock to another person.
            for box in others:
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 2)
            self.draw_tracked_objects(img, tracked_objects)
            if self.SHOW_STATUS_BADGE:
                self.draw_status_badge(img)
        else:
            # The night / thermal colouring is done by the panel writer
            # thread (fb_view.filter), so web and RTP show the HUD over the
            # normal picture and the pipeline does not pay for it.
            locked = None
            if self.state.startswith("LOCKED") and tracked_objects:
                obj = tracked_objects[0]
                det = getattr(obj, "last_detection", None)
                score = float(det.scores[0]) if det is not None and det.scores is not None else None
                locked = (self.box_of(obj), obj.id, score)
            sc = self.servo_controller
            try:
                self.hud.draw(
                    img, style, self.frame_count, locked, others, self.state,
                    sc.pan_angle if sc else 0.0, sc.tilt_angle if sc else 0.0,
                    self.ui is not None and self.ui.manual,
                    bool(sc and sc.searching),
                )
            except Exception as exc:
                # Decoration only: never let a drawing bug stop tracking.
                print("[WARNING] HUD draw failed: %s" % exc, flush=True)

        if self.fire_frame is not None:
            try:
                self.hud.draw_fire(
                    img, self.frame_count - self.fire_frame, self.fire_hit,
                    self.fire_point, self.score,
                )
            except Exception as exc:
                print("[WARNING] fire effect failed: %s" % exc, flush=True)

        # Where the last tap landed, for about half a second, so a tap that
        # misses the box is visible on the panel as a miss.
        if self.touch_marker is not None:
            (mx, my), f0 = self.touch_marker
            if self.frame_count - f0 <= 8:
                cv2.circle(img, (int(mx), int(my)), 28, (255, 220, 0), 4)
            else:
                self.touch_marker = None

        if self.web_view:
            self.web_view.publish(img)
        if self.fb_view:
            self.fb_view.publish(img)

        if self.debug:
            self.debug.log(self.debug_str)
            self.debug_str = ""

        return img

    # ----------------------------------------------------------------------
    # Target selection and the lock / hybrid modes
    # ----------------------------------------------------------------------

    @staticmethod
    def box_of(obj):
        """Flatten a Norfair estimate into [x1, y1, x2, y2] floats."""
        p = obj.estimate
        return [float(p[0][0]), float(p[0][1]), float(p[1][0]), float(p[1][1])]

    @staticmethod
    def box_center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @staticmethod
    def iou(a, b):
        """Intersection over union of two [x1, y1, x2, y2] boxes."""
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def pick_target(self, candidates, frame_w, frame_h):
        """
        Choose one candidate to lock onto, following self.lock_policy.

        candidates is a list of (key, box) pairs, where key is whatever the
        caller wants back: a Norfair object for lock mode, an index for
        hybrid. Returns the winning pair, or None when there is nothing to
        pick.
        """
        if not candidates:
            return None

        if self.lock_policy == "first":
            return candidates[0]

        if self.lock_policy == "largest":
            return max(
                candidates,
                key=lambda kb: max(0.0, kb[1][2] - kb[1][0])
                * max(0.0, kb[1][3] - kb[1][1]),
            )

        # closest_to_center, the default: least work for a pan/tilt rig.
        fcx, fcy = frame_w / 2.0, frame_h / 2.0
        return min(
            candidates,
            key=lambda kb: (self.box_center(kb[1])[0] - fcx) ** 2
            + (self.box_center(kb[1])[1] - fcy) ** 2,
        )

    def update_lock(self, tracked_objects, img):
        """
        Reduce the tracker output to the single locked target.

        Detection keeps running on every frame; this only decides which of the
        tracked objects is "the" target. The lock survives an id rename
        because re-acquisition matches on position, not on id: when the locked
        id disappears, whatever is nearest the last known box within
        LOCK_REACQUIRE_RADIUS inherits the lock.

        Returns a list with zero or one Norfair object, so every stage after
        this point (positions, printing, drawing) stays unchanged.
        """
        frame_h, frame_w = img.shape[0], img.shape[1]

        by_id = {obj.id: obj for obj in tracked_objects}

        # A tap on the panel since the last frame. On an object: lock onto
        # it, whatever lock_policy would have picked. Off every object: drop
        # the lock and its memory, so the automatic pick below starts over.
        req, self.touch_request = self.touch_request, None
        if req is not None:
            tx, ty = req
            hits = [
                (obj, box)
                for obj, box in ((o, self.box_of(o)) for o in tracked_objects)
                if box[0] <= tx <= box[2] and box[1] <= ty <= box[3]
            ]
            if hits:
                # The smallest box under the finger, a face in front of a body.
                obj, box = min(
                    hits, key=lambda ob: (ob[1][2] - ob[1][0]) * (ob[1][3] - ob[1][1])
                )
                print("[f:%d] * touch lock id=%d" % (self.frame_count, obj.id))
                if self.ui is not None:
                    self.ui.manual = False
                self.locked_id = obj.id
                self.locked_box = box
                self.lock_lost_frame = None
                self.state = "LOCKED id=%d" % obj.id
                return [obj]
            print(
                "[f:%d] - touch release (id=%s) at (%d,%d), boxes %s"
                % (self.frame_count, self.locked_id, tx, ty,
                   [[int(v) for v in b] for _, b in
                    ((o, self.box_of(o)) for o in tracked_objects)])
            )
            self.locked_id = None
            self.locked_box = None
            self.lock_lost_frame = None

        # Still holding the same id: nothing to decide.
        if self.locked_id is not None and self.locked_id in by_id:
            obj = by_id[self.locked_id]
            self.locked_box = self.box_of(obj)
            self.state = "LOCKED id=%d" % obj.id
            return [obj]

        candidates = [(obj, self.box_of(obj)) for obj in tracked_objects]

        # Lost the id. Try to inherit the lock from whatever is nearest the
        # last known position before falling back to a fresh pick.
        recently_lost = (
            self.lock_lost_frame is not None
            and self.frame_count - self.lock_lost_frame <= self.LOCK_MEMORY_FRAMES
        )
        if (
            (self.locked_id is not None or recently_lost)
            and self.locked_box is not None
            and candidates
        ):
            lcx, lcy = self.box_center(self.locked_box)
            diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
            limit = self.LOCK_REACQUIRE_RADIUS * diag

            nearest, best = None, None
            for obj, box in candidates:
                cx, cy = self.box_center(box)
                d = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
                if best is None or d < best:
                    nearest, best = (obj, box), d

            if nearest is not None and best <= limit:
                obj, box = nearest
                print(
                    "[f:%d] ~ lock handed over id=%s -> id=%d (%.0f px)"
                    % (self.frame_count, self.locked_id, obj.id, best)
                )
                self.locked_id = obj.id
                self.locked_box = box
                self.state = "LOCKED id=%d" % obj.id
                return [obj]

        if not candidates:
            if self.locked_id is not None:
                print("[f:%d] - lock lost (id=%s)"
                      % (self.frame_count, self.locked_id))
                self.locked_id = None
                self.lock_lost_frame = self.frame_count
            self.state = "SEARCHING"
            return []

        obj, box = self.pick_target(candidates, frame_w, frame_h)
        if obj.id != self.locked_id:
            print("[f:%d] * locked id=%d" % (self.frame_count, obj.id))
        self.locked_id = obj.id
        self.locked_box = box
        self.state = "LOCKED id=%d" % obj.id
        return [obj]

    # ----------------------------------------------------------------------
    # hybrid mode: visual tracking with periodic detector corrections
    # ----------------------------------------------------------------------

    def get_cv_tracker_factory(self):
        """
        Find the configured OpenCV visual tracker and return (factory, name).

        OpenCV moved these constructors between the cv2 and cv2.legacy
        namespaces across versions, and which ones exist depends on how the
        build was configured, so probe rather than assume. The order is the
        configured algorithm first, then the rest as a fallback, so a build
        without the preferred one still starts instead of dying.

        See TRACKER_ALGO for why the default is MOSSE and not the more
        accurate CSRT.
        """
        wanted = str(getattr(self.model, "tracker_algo", self.TRACKER_ALGO)).lower()
        known = {
            "mosse": "TrackerMOSSE",
            "kcf": "TrackerKCF",
            "csrt": "TrackerCSRT",
        }
        if wanted not in known:
            print(
                "[WARNING] Unknown tracker_algo '%s', using '%s'. Valid: %s."
                % (wanted, self.TRACKER_ALGO, ", ".join(sorted(known)))
            )
            wanted = self.TRACKER_ALGO

        order = [known[wanted]] + [v for k, v in known.items() if k != wanted]
        namespaces = [getattr(cv2, "legacy", None), cv2]

        for base in order:
            for ns in namespaces:
                if ns is None:
                    continue
                for attr in (base + "_create", base):
                    factory = getattr(ns, attr, None)
                    if callable(factory):
                        if base != known[wanted]:
                            print(
                                "[WARNING] %s is not in this OpenCV build, "
                                "falling back to %s." % (known[wanted], base)
                            )
                        print(
                            "[INFO] Hybrid mode: OpenCV %s at scale %.2f."
                            % (base, self.tracker_scale)
                        )
                        return factory, base

        print(
            "[ERROR] track_mode: hybrid needs an OpenCV visual tracker "
            "(MOSSE/KCF/CSRT) and this OpenCV build has none. "
            "Install opencv-contrib, or use track_mode: lock."
        )
        sys.exit()

    def to_track_frame(self, img):
        """
        Downscaled copy of the frame for the visual tracker, or the frame
        itself when the scale is 1. See TRACKER_SCALE for why this is worth a
        resize.
        """
        if self.tracker_scale >= 0.999:
            return img
        return cv2.resize(
            img,
            (
                max(2, int(img.shape[1] * self.tracker_scale)),
                max(2, int(img.shape[0] * self.tracker_scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    def detection_boxes(self, bbox):
        """Detections of the tracked classes as plain [x1, y1, x2, y2] boxes."""
        return [
            [float(d.points[0][0]), float(d.points[0][1]),
             float(d.points[1][0]), float(d.points[1][1])]
            for d in self.yolo_detections_to_norfair_detections(bbox)
        ]

    def start_visual_track(self, img, box):
        """
        Point a fresh visual tracker at box, given in full frame pixels.

        The tracker runs on a downscaled copy, so the box is scaled down on
        the way in; update_hybrid scales its answer back up. Returns True when
        the tracker took the box.
        """
        frame_h, frame_w = img.shape[0], img.shape[1]

        x1 = max(0, min(int(box[0]), frame_w - 2))
        y1 = max(0, min(int(box[1]), frame_h - 2))
        x2 = max(x1 + 1, min(int(box[2]), frame_w - 1))
        y2 = max(y1 + 1, min(int(box[3]), frame_h - 1))

        small = self.to_track_frame(img)
        s = self.tracker_scale
        sx1, sy1 = int(x1 * s), int(y1 * s)
        sw = max(2, int((x2 - x1) * s))
        sh = max(2, int((y2 - y1) * s))
        # Keep the scaled box inside the scaled frame, otherwise init throws.
        sx1 = max(0, min(sx1, small.shape[1] - sw - 1))
        sy1 = max(0, min(sy1, small.shape[0] - sh - 1))

        try:
            tracker = self._cv_tracker_factory()
            tracker.init(small, (sx1, sy1, sw, sh))
        except Exception as exc:
            # A failed init must not take the pipeline thread down with it:
            # this runs inside the streaming thread, and an exception there
            # kills the flow while the process stays up, which looks like a
            # hang rather than a crash.
            print("[WARNING] Visual tracker init failed: %s" % exc)
            self.cv_tracker = None
            return False

        self.cv_tracker = tracker
        self.locked_box = [float(x1), float(y1), float(x2), float(y2)]
        self.frames_since_redetect = 0
        return True

    def hybrid_acquire(self, img, bbox, frame_w, frame_h):
        """Run the detector, pick a target, hand it to the visual tracker."""
        boxes = self.detection_boxes(bbox)
        candidates = list(enumerate(boxes))

        picked = self.pick_target(candidates, frame_w, frame_h)
        if picked is None:
            self.state = "SEARCHING"
            return []

        _, box = picked
        if not self.start_visual_track(img, box):
            self.state = "SEARCHING"
            return []

        self.locked_id = (self.locked_id or 0) + 1
        self.hybrid_age = 0
        print("[f:%d] * target acquired id=%d (handed to visual tracking)"
              % (self.frame_count, self.locked_id))
        self.state = "TRACKING"
        return [VisualTrack(self.locked_box, self.locked_id, self.hybrid_age)]

    def hybrid_correct(self, img, bbox, current_box):
        """
        Pull the visual tracker back onto the detector's idea of the target.

        A visual tracker accumulates drift: it follows whatever the patch
        looked like, so over seconds it slides off the object, and once it is
        off there is nothing to pull it back. Re-seeding it from a detection
        that still overlaps the tracked box fixes that without giving up the
        target. When nothing overlaps enough, the tracker is left alone: a
        detector miss is common and is not by itself proof the track is wrong.
        """
        boxes = self.detection_boxes(bbox)
        if not boxes:
            return None

        best_box, best_iou = None, 0.0
        for box in boxes:
            score = self.iou(current_box, box)
            if score > best_iou:
                best_box, best_iou = box, score

        if best_box is None or best_iou < self.REDETECT_MIN_IOU:
            return None

        self.start_visual_track(img, best_box)
        return best_box

    def update_hybrid(self, img, bbox):
        """
        One frame of hybrid mode.

        Returns a list with zero or one VisualTrack so the rest of __call__
        does not need to know which mode produced it.
        """
        frame_h, frame_w = img.shape[0], img.shape[1]

        if self.cv_tracker is None:
            return self.hybrid_acquire(img, bbox, frame_w, frame_h)

        ok, rect = self.cv_tracker.update(self.to_track_frame(img))
        if not ok:
            print("[f:%d] - visual tracking lost the target (id=%s)"
                  % (self.frame_count, self.locked_id))
            self.cv_tracker = None
            return self.hybrid_acquire(img, bbox, frame_w, frame_h)

        # Back to full frame pixels: everything downstream, including the
        # dx/dy the controller reads, works in the drawing frame's scale.
        inv = 1.0 / self.tracker_scale
        box = [float(rect[0]) * inv, float(rect[1]) * inv,
               float(rect[0] + rect[2]) * inv, float(rect[1] + rect[3]) * inv]
        self.locked_box = box
        self.hybrid_age = getattr(self, "hybrid_age", 0) + 1
        self.frames_since_redetect += 1
        self.state = "TRACKING"

        if (
            self.redetect_interval
            and self.frames_since_redetect >= self.redetect_interval
        ):
            self.frames_since_redetect = 0
            corrected = self.hybrid_correct(img, bbox, box)
            self.state = (
                "CORRECTED" if corrected is not None else "TRACKING (no correction)"
            )
            if corrected is not None:
                box = corrected

        return [VisualTrack(box, self.locked_id, self.hybrid_age)]

    def camera_motion(self, frame_w, frame_h):
        """
        Norfair coordinate transformation for where the camera pointed when
        this frame was taken, or None without servos.

        The sign follows the controller: moving an axis the way update() does
        for a target on the positive side brings that target towards the
        centre, i.e. shifts the picture towards negative x (or y). So the
        world position is the image position plus angle * sign * px/degree,
        and Norfair wants movement_vector = relative - absolute.
        """
        sc = self.servo_controller
        if not self.motion_compensation or not sc or not sc.enabled:
            return None
        if self.angle_history:
            pan, tilt = self.angle_history[0]
        else:
            pan, tilt = sc.pan_angle, sc.tilt_angle
        sign_pan = 1.0 if sc.pan_invert else -1.0
        sign_tilt = 1.0 if sc.tilt_invert else -1.0
        px_deg_x = frame_w / (2.0 * sc.half_hfov)
        px_deg_y = frame_h / (2.0 * sc.half_vfov)
        return self.TranslationTransformation(
            np.array(
                [-pan * sign_pan * px_deg_x, -tilt * sign_tilt * px_deg_y]
            )
        )

    def debug_servo_line(self, dy):
        sc = self.servo_controller
        p = self.tracked_positions[0] if self.tracked_positions else None
        score = None
        if p is not None and self._live_detections is not None:
            for obj in self._last_tracked:
                if obj.id == p["id"] and obj.last_detection is not None:
                    score = float(obj.last_detection.scores[0])
        print(
            "[SERVO] f=%d %s live=%s score=%s dx=%s dy=%s pan=%.1f tilt=%.1f"
            % (
                self.frame_count,
                self.state,
                None if p is None else p["live"],
                "-" if score is None else "%.2f" % score,
                "-" if p is None else "%+.2f" % p["dx"],
                "-" if dy is None else "%+.2f" % dy,
                sc.pan_angle,
                sc.tilt_angle,
            ),
            flush=True,
        )

    def update_pan_tilt(self, frame_w, frame_h):
        """
        Drive the pan-tilt mechanism towards the tracked/locked target.
        - When LOCKED: Smoothly track the target.
        - When LOCKED but the target is only predicted this frame: hold still.
        - When SEARCHING (target lost): hold for servo_search_delay, then sweep
          the pan axis until something is locked again (see search()).
        """
        if not self.servo_controller or not self.servo_controller.enabled:
            return

        # Recoil after a shot runs on top of whatever mode is active.
        self.servo_controller.step_recoil()

        if self.ui is not None and self.ui.manual:
            # MANUAL: the D-pad drives the servos, nothing else does.
            import time

            now = time.time()
            dt = 0.0 if self._manual_time is None else min(0.2, now - self._manual_time)
            self._manual_time = now
            right, up = self.ui.move
            if right or up:
                step = self.manual_speed * dt
                self.servo_controller.nudge(right * step, up * step)
            self.servo_controller.searching = False
            # Returning to AUTO should look around from here, not jump
            # straight into a sweep.
            self.servo_controller.last_target_time = now
            self.servo_controller.reset_filters()
            return
        self._manual_time = None

        is_locked = self.state.startswith("LOCKED")
        dy = None

        if is_locked and self.tracked_positions and not self.tracked_positions[0]["live"]:
            # Locked, but the target was not detected in this frame and its box
            # is only the Kalman prediction. Hold still instead of steering
            # after it: a prediction keeps its last velocity, and with the
            # camera itself moving it ran off the frame edge (dx -1.42 in the
            # log) and swung the rig away from where the target reappeared.
            # The filters are kept, this is a brief occlusion, not a new
            # target.
            self.servo_controller.hold()
        elif is_locked and self.tracked_positions:
            target = self.tracked_positions[0]
            x1, y1, x2, y2 = target["bbox"]
            margin = 0.02 * frame_h
            if y1 <= margin and y2 >= frame_h - margin:
                # The box runs off the top AND the bottom of the frame: the
                # person is close enough to fill it, and where servo_aim_y
                # lands in the real body is unknown. Aiming at it anyway
                # always reads "above centre", since the box never shrinks
                # however far the camera tilts, and drove tilt into its
                # end stop. Leave tilt alone until the box fits.
                dy = 0.0
            else:
                aim_y = y1 + self.servo_aim_y * (y2 - y1)
                dy = (aim_y - frame_h / 2.0) / (frame_h / 2.0)
            self.servo_controller.update(target["dx"], dy)
        else:
            # No target: hold briefly where it was lost, then sweep for it.
            self.servo_controller.search()
            self.servo_controller.reset_filters()

        if self.servo_debug and is_locked:
            self.debug_servo_line(dy)

    def draw_status_badge(self, frame):
        """
        Draw the current state in the top left corner.

        The state text alone says what the flow is doing, so the mode name is
        not repeated here: "LOCKED id=3" can only come from lock, and
        "TRACKING: 3 objects" only from multi. Which mode is configured is printed
        once at startup instead, where it belongs.
        """
        label = self.state
        if self.ui is not None and self.ui.manual:
            label = "MANUEL | " + label
        if self.servo_controller and self.servo_controller.searching:
            label += " (scan)"
        if self.servo_controller and self.servo_controller.enabled:
            label += " | Pan:%+.0f Tilt:%+.0f" % (
                self.servo_controller.pan_angle,
                self.servo_controller.tilt_angle,
            )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_size, thickness
        )

        # Green while a target is held, blue while searching, so the state is
        # readable from across the room without reading the text.
        holding = not self.state.startswith("SEARCHING")
        color = (0, 170, 0) if holding else (0, 140, 220)

        x, y = 8, 8
        cv2.rectangle(
            frame,
            (x, y),
            (x + text_w + 12, y + text_h + baseline + 8),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x + 6, y + text_h + 4),
            font,
            font_size,
            (255, 255, 255),
            thickness,
        )

    def update_tracked_positions(self, tracked_objects, frame_w, frame_h):
        """
        Refresh self.tracked_positions from the current tracker state.

        For every tracked object the box centre is expressed as an offset from
        the centre of the frame, normalized so the value does not depend on the
        resolution:

            cx = (x1 + x2) / 2 ;  cy = (y1 + y2) / 2
            dx = (cx - W/2) / (W/2)
            dy = (cy - H/2) / (H/2)

        W and H are the size of the drawing frame (1280x720 with the shipped
        people_tracking.yaml), not the size of the model input.

            dx = -1.0 left edge,  0.0 horizontally centred, +1.0 right edge
            dy = -1.0 top edge,   0.0 vertically centred,   +1.0 bottom edge

        IMPORTANT, THE SIGN OF dy: the image y axis grows DOWNWARDS, so a
        POSITIVE dy means the object is BELOW the centre of the frame, not
        above it. dx and dy are meant to be used as the error signal of a servo
        pan/tilt loop later on, where getting the sign wrong makes the loop run
        away from the target instead of towards it. If the tilt servo treats
        "up" as positive, negate dy when feeding it.

        The values come from the tracker estimate rather than the raw
        detection, and are deliberately NOT clamped to [-1, 1]: while an object
        is occluded the predicted box can drift past the edge of the frame, and
        a consumer is better off seeing that than seeing a value pinned at the
        border. draw_tracked_objects() clamps separately, for drawing only.

        Args:
            tracked_objects: Norfair TrackedObject list from tracker.update()
            frame_w (int): Width of the drawing frame in pixels
            frame_h (int): Height of the drawing frame in pixels
        """
        half_w = frame_w / 2.0
        half_h = frame_h / 2.0

        positions = []
        for obj in tracked_objects:
            points = obj.estimate
            x1, y1 = float(points[0][0]), float(points[0][1])
            x2, y2 = float(points[1][0]), float(points[1][1])

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            live = self._live_detections is None or (
                id(getattr(obj, "last_detection", None)) in self._live_detections
            )

            positions.append(
                {
                    "id": int(obj.id),
                    "live": live,
                    "dx": (cx - half_w) / half_w,
                    "dy": (cy - half_h) / half_h,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "cx": int(cx),
                    "cy": int(cy),
                    "age": int(obj.age),
                }
            )

        self.tracked_positions = positions

    def report_tracked_positions(self):
        """
        Print the current tracker state, keeping the terminal readable.

        Ids that appeared or disappeared are printed on the frame it happened,
        the full list of active objects only every PRINT_EVERY_N_FRAMES frames.
        Nothing at all is printed while no object is being tracked.
        """
        current_ids = set(p["id"] for p in self.tracked_positions)

        appeared = current_ids - self._reported_ids
        lost = self._reported_ids - current_ids
        self._reported_ids = current_ids

        # The ncurses performance table owns the terminal unless the app was
        # started with -n/--no-curses, and printing over it corrupts it.
        if not self.PRINT_EVERY_N_FRAMES or utils.curses_active:
            return

        for obj_id in sorted(appeared):
            print("[f:%d] + id=%d" % (self.frame_count, obj_id), flush=True)
        for obj_id in sorted(lost):
            print("[f:%d] - id=%d" % (self.frame_count, obj_id), flush=True)

        if not self.tracked_positions:
            return
        if self.frame_count % self.PRINT_EVERY_N_FRAMES:
            return

        for p in self.tracked_positions:
            print(
                "[f:%d] id=%d  dx=%+.2f dy=%+.2f  px=(%d,%d) age=%d"
                % (
                    self.frame_count,
                    p["id"],
                    p["dx"],
                    p["dy"],
                    p["cx"],
                    p["cy"],
                    p["age"],
                ),
                flush=True,
            )

    def draw_tracked_objects(self, frame, tracked_objects):
        """
        Draw a bounding box and an "id=N" label for every tracked object.

        The tracker estimate, not the raw detection, is drawn: it is what
        survives a missed frame, so the box stays put while an object is
        briefly occluded.

        Args:
            frame (numpy array): Frame to draw on, modified in place
            tracked_objects: Norfair TrackedObject list from tracker.update()
        """
        frame_h, frame_w = frame.shape[0], frame.shape[1]

        for obj in tracked_objects:
            # estimate is the Kalman filtered version of the two corner points
            # the detection was built from: [[x1, y1], [x2, y2]].
            points = obj.estimate
            x1, y1 = int(points[0][0]), int(points[0][1])
            x2, y2 = int(points[1][0]), int(points[1][1])

            # A predicted box can drift off frame while an object is occluded.
            x1 = max(0, min(x1, frame_w - 1))
            x2 = max(0, min(x2, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            y2 = max(0, min(y2, frame_h - 1))

            color = self.Palette.choose_color(obj.id)
            luma = ((66 * color[0] + 129 * color[1] + 25 * color[2] + 128) >> 8) + 16
            text_color = (0, 0, 0) if luma >= 128 else (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = "id=%d" % obj.id
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_size, thickness
            )

            # Label sits above the box, or just inside it when the box is
            # already touching the top of the frame.
            label_bottom = y1 - baseline
            if label_bottom - text_h < 0:
                label_bottom = y1 + text_h + baseline

            cv2.rectangle(
                frame,
                (x1, label_bottom - text_h - baseline),
                (x1 + text_w + 6, label_bottom + baseline),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 3, label_bottom),
                font,
                font_size,
                text_color,
                thickness,
            )

    def yolo_detections_to_norfair_detections(self, results_bbox) -> List["Detection"]:
        """convert detections_as_xywh to norfair detections"""
        norfair_detections: List["Detection"] = []
        for b in results_bbox:
            if b[5] > self.model.viz_threshold and (
                self.track_class_indices is None
                or int(b[4]) in self.track_class_indices
            ):
                bbox = np.array(
                    [
                        [b[0].item(), b[1].item()],
                        [b[2].item(), b[3].item()],
                    ]
                )
                scores = np.array([b[5], b[5]])
                norfair_detections.append(
                    self.Detection(points=bbox, scores=scores, label=int(b[4]))
                )

                if self.debug:
                    # Was hardcoded to "person"; the tracked class is now
                    # configurable, so log the class actually detected.
                    self.debug_str += (
                        "class %d " % int(b[4]) + str(bbox.flatten()) + "\n"
                    )

        return norfair_detections