from .base.base import BaseMotionDetector
from .base.tools import (
    compute_flow,
    get_motion_mask,
    get_contour_detections_angle,
    non_max_suppression,
)
import numpy as np
import cv2

class OFMD(BaseMotionDetector):
    """OpticalFlowMotionDetection"""
    def __init__(self, 
                 mask_kernel_size=(3, 3),
                 bbox_thresh=400,
                 nms_thresh=0.1, 
                 motion_thresh=1):
        super().__init__()
        self.mask_kernel_size = mask_kernel_size
        self.bbox_thresh = bbox_thresh
        self.nms_thresh = nms_thresh
        self.motion_thresh = motion_thresh

        self.mask_kernel = np.ones(mask_kernel_size, dtype=np.uint8)
        self.frame1=None
        self.frame2=None
        self._image = None

    @property
    def mask(self):
        flow = compute_flow(self.frame1, self.frame2)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mask = get_motion_mask(mag, self.motion_thresh, self.mask_kernel)
        return motion_mask
    
    @property
    def boxes(self):
        flow = compute_flow(self.frame1, self.frame2)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mask = get_motion_mask(mag, self.motion_thresh, self.mask_kernel)
        detections = get_contour_detections_angle(motion_mask, ang)
        if len(detections) > 0:
            # Separate bounding boxes and scores
            bboxes = detections[:, :4]
            scores = detections[:, -1]

            # Perform Non-Maximal Suppression
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