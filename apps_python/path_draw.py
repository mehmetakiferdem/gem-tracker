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
from typing import Callable, Optional, Sequence, Tuple
import numpy as np
import cv2
import math
from norfair.tracker import TrackedObject
from norfair.drawing.color import Palette


class PathDraw:
    """
    """
    
    def __init__(
                self,
                history_size: int = 30,
                thickness: Optional[int] = 2,
                radius: Optional[int] = 10,
                attenuation: float = 0.01,
        ):
        
        self.history_size = history_size
        self.thickness = thickness
        self.radius = radius
        self.attenuation = attenuation

        self.path_list = dict()

    def draw(
            self,
            frame,
            tracked_objects: Sequence[TrackedObject]
    )-> np.array:
        
        list_id = []
        # update path list
        for obj in tracked_objects:
            list_id.append(obj.id)
            if obj.id in self.path_list:
                self.path_list[obj.id].add(obj.estimate)
            else:
                self.path_list[obj.id] = PathList(self.history_size, obj.estimate)

        # keep only new or updated objects path and remove the rest.
        self.path_list = {key: self.path_list[key] 
            for key in self.path_list if key in list_id}
        return self.draw_circle(frame)

    def draw_circle(self, frame):
        for key in self.path_list:
            radius_factor = 1
            for point in self.path_list[key].path:
                color = Palette.choose_color(key)
                radius = math.ceil(self.radius * radius_factor)
                cv2.circle(
                    frame,
                    tuple(point.astype(int)),
                    radius=radius,
                    color=color,
                    thickness=self.thickness)
                radius_factor = radius_factor * 0.95
        return frame

from queue import Queue
class PathList:
    """
    """
    
    def __init__(self, size, bbox):
        self.size = size
        self.path = [] 
        self.len = 0

        self.add(bbox)

    def add(self, bbox):
        # calcualte center point
        center = np.mean(np.array(bbox), axis=0)
        self.path.insert(0, center)
        if len(self.path) > self.size:
            self.path.pop(self.size)

        


