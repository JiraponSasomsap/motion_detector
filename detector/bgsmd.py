from .base.base import BaseMotionDetector
from .base.tools import (
    get_motion_mask_morphologyEx,
    get_contour_detections,
    non_max_suppression
)

import numpy as np
import cv2

class BGSMD(BaseMotionDetector):
    """BackgroundSubtractionMotionDetection"""
    def __init__(self, 
                 mask_kernel_size=(3,3), 
                 bbox_thresh=400, 
                 nms_thresh=0.1, 
                 sub_type='MOG2', 
                 **kwargs):
        super().__init__()
        self.mask_kernel_size = mask_kernel_size
        self.bbox_thresh = bbox_thresh
        self.nms_thresh = nms_thresh
        self.sub_type = sub_type

        self.mask_kernel = np.ones(mask_kernel_size, dtype=np.uint8)
        self.frame1=None
        self.frame2=None
        self._image = None

        if sub_type == "MOG2":
            # self.backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=True)
            self.backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=kwargs.get('varThreshold', 16), 
                                                              detectShadows=kwargs.get('detectShadows', True),
                                                              history=kwargs.get('history', 0), 
                                                              )
            self.backSub.setShadowThreshold(0.5)
        else:
            self.backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=kwargs.get('dist2Threshold', 1000), 
                                                             detectShadows=kwargs.get('detectShadows', True),
                                                             history=kwargs.get('history', 0), 
                                                             )
        self.i_frame = 0

    @property
    def mask(self):
        if self.i_frame == 0:
            fg_mask = self.backSub.apply(self.frame1)
            fg_mask = self.backSub.apply(self.frame2) 
            self.i_frame += 2
        else:
            fg_mask = self.backSub.apply(self.frame2)    
            self.i_frame += 1 
                    
        motion_mask = get_motion_mask_morphologyEx(fg_mask, kernel=self.mask_kernel)
        return motion_mask
    
    @property
    def boxes(self):
        detections = get_contour_detections(self.mask, self.bbox_thresh)
        if len(detections)> 0:
            # separate bboxes and scores
            bboxes = detections[:, :4]
            scores = detections[:, -1]

            # perform Non-Maximal Supression on initial detections
            return np.array(non_max_suppression(bboxes, scores, self.nms_thresh), dtype=np.int32)
        else:
            return np.array(detections, dtype=np.int32)
    
    @property
    def image(self):
        return self._image

    def __call__(self, frame1, frame2):
        self._image = frame1.copy()
        self.frame1 = frame1.copy()
        self.frame2 = frame2.copy()
        return self