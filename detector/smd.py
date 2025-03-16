from .base.base import BaseMotionDetector
from .base.tools import get_mask, get_contour_detections, non_max_suppression
import numpy as np

class SMD(BaseMotionDetector):
    """SimpleMotionDetection"""

    def __init__(self, 
                 mask_kernel_size=(3,3), 
                 bbox_thresh=400, 
                 nms_thresh=0.1):
        super().__init__()
        self.mask_kernel_size = mask_kernel_size
        self.bbox_thresh = bbox_thresh
        self.nms_thresh = nms_thresh

        self.mask_kernel = np.ones(mask_kernel_size, dtype=np.uint8)
        self.frame1=None
        self.frame2=None
        self._image = None

    @property
    def mask(self):
        return get_mask(frame1=self.frame1,
                        frame2=self.frame2,
                        mask_kernel_size=self.mask_kernel_size)
    
    @property
    def boxes(self):
        detections = get_contour_detections(self.mask, self.bbox_thresh)
        if len(detections) == 0:
            return []
        bboxes = detections[:, :4]
        scores = detections[:, -1]
        return non_max_suppression(bboxes, scores, self.nms_thresh)
    
    @property
    def image(self):
        return self._image

    def __call__(self, frame1, frame2):
        self._image = frame1.copy()
        self.frame1 = frame1.copy()
        self.frame2 = frame2.copy()
        return self