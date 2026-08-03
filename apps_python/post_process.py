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
            bbox[..., (0, 2)] /= self.model.resize[0]
            bbox[..., (1, 3)] /= self.model.resize[1]

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
            bbox[..., (0, 2)] /= self.model.resize[0]
            bbox[..., (1, 3)] /= self.model.resize[1]

        # ------------------------------------------------------------------
        # OBJECT TRACKING
        # ------------------------------------------------------------------
        # PostProcessDetection scales normalized boxes to pixels inside
        # overlay_bounding_box(). The tracker, heat map and time counter all
        # work in absolute frame pixels, so scale up once here instead.
        bbox[..., (0, 2)] *= img.shape[1]
        bbox[..., (1, 3)] *= img.shape[0]

        self.frame_count += 1

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
            tracked_objects = self.tracker.update(detections=detections)
            if self.track_mode == self.MODE_LOCK:
                tracked_objects = self.update_lock(tracked_objects, img)
            else:
                self.state = (
                    "TRACKING: %d objects" % len(tracked_objects)
                    if tracked_objects
                    else "SEARCHING"
                )

        self.update_tracked_positions(tracked_objects, img.shape[1], img.shape[0])
        self.report_tracked_positions()

        img = self.pathd.draw(img, tracked_objects)
        self.draw_tracked_objects(img, tracked_objects)

        if self.SHOW_STATUS_BADGE:
            self.draw_status_badge(img)

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

        # Still holding the same id: nothing to decide.
        if self.locked_id is not None and self.locked_id in by_id:
            obj = by_id[self.locked_id]
            self.locked_box = self.box_of(obj)
            self.state = "LOCKED id=%d" % obj.id
            return [obj]

        candidates = [(obj, self.box_of(obj)) for obj in tracked_objects]

        # Lost the id. Try to inherit the lock from whatever is nearest the
        # last known position before falling back to a fresh pick.
        if self.locked_id is not None and self.locked_box is not None and candidates:
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

    def draw_status_badge(self, frame):
        """
        Draw the current state in the top left corner.

        The state text alone says what the flow is doing, so the mode name is
        not repeated here: "LOCKED id=3" can only come from lock, and
        "TRACKING: 3 objects" only from multi. Which mode is configured is printed
        once at startup instead, where it belongs.
        """
        label = self.state

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

            positions.append(
                {
                    "id": int(obj.id),
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