from .base.base import BaseMotionDetector
from .base.tools import (merge_boxes, 
                         get_bounding_boxes, 
                         is_daytime_histogram)
import cv2
import numpy as np

class FrameDiff(BaseMotionDetector):
    def __init__(self, 
                 valThresh=125, 
                 boxAreaThresh=0.02,             
                 image_preprocess_func=None, 
                 image_postprocess_func=None):
        super().__init__(image_preprocess_func, 
                         image_postprocess_func)
        self.valThresh = valThresh
        self.boxAreaThresh = boxAreaThresh

    @property
    def immask(self):
        power_up = 1.5 if is_daytime_histogram(self.frame1) else 1.2
        mask = cv2.absdiff(self.frame1, self.frame2) ** power_up
        immask = mask.astype(np.uint8)

        immask_thresh = np.zeros_like(immask, dtype=np.uint8)
        immask_thresh[immask >= self.valThresh] = 255
        return immask_thresh 
    
    @property
    def boxes(self):
        
        h, w = self.frame1.shape[:2]
        h_ths = h*self.boxAreaThresh
        w_ths = w*self.boxAreaThresh

        bounding_boxes = get_bounding_boxes(self.immask)

        boxes = []

        for box in bounding_boxes:
            x=abs(box[0]-box[2])
            y=abs(box[1]-box[3])
            area = x*y
            if area > h_ths*w_ths:
                boxes.append(box)

        if len(boxes) == 0:
            return np.array([])

        padding = np.array([w*self.boxAreaThresh, h*self.boxAreaThresh])

        boxes = merge_boxes(boxes, padding)
        boxes = merge_boxes(boxes, padding) # recheck 
        
        boxes = np.array(boxes, dtype=np.int32)

        boxes[:, 0:2] = np.maximum(boxes[:, 0:2], 0)  # Clip x1, y1 to ≥ 0
        boxes[:, 2] = np.minimum(boxes[:, 2], self.frame1.shape[1])  # Clip x2 to ≤ width
        boxes[:, 3] = np.minimum(boxes[:, 3], self.frame1.shape[0])  # Clip y2 to ≤ height
        return boxes