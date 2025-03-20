from abc import ABC, abstractmethod
import cv2
import numpy as np 
from .basetools import (ignore_areas, 
                        interested_areas,
                        conv_boxes_to_boxes_norm,)
from typing import Union, List, Tuple

class _GetResults:
    def __init__(self, base:'BaseMotionDetector'):
        self.__base = base
    
    @property
    def boxes_n(self) -> List[Tuple[float, float, float, float]]:
        boxes_norm = conv_boxes_to_boxes_norm(self.__base.boxes)
        return boxes_norm

    @property
    def frame1(self) -> cv2.typing.MatLike:
        return self.__base.frame1
    
    @property
    def frame2(self) -> cv2.typing.MatLike:
        return self.__base.frame2

    @property
    def boxes(self) -> List[Tuple[int, int, int, int]]:
        return self.__base.boxes

    @property
    def immask(self):
        if self.__base.image_postprocess_func is None:
            return self.__base.immask
        return self.__base.image_postprocess_func(self.__base.immask)

    @property
    def imshape(self):
        if self.__base.frame1 is None:
            return None
        return self.__base.frame1.shape
    
    def boxes_cfg(self, padding=(0,0)):
        new_boxes = []
        for box in self.boxes:
            # Apply padding
            box[0] -= padding[0]
            box[1] -= padding[1]
            box[2] += padding[0]
            box[3] += padding[1]

            # Ensure the coordinates are within bounds
            box[0] = max(0, box[0])  # Ensure x1 is not negative
            box[1] = max(0, box[1])  # Ensure y1 is not negative
            box[2] = min(self.imshape[1], box[2])  # Ensure x2 is within image width
            box[3] = min(self.imshape[0], box[3])  # Ensure y2 is within image height
            new_boxes.append(box)
        return np.array(new_boxes)
    
    @property
    def imcrops(self):
        return self.imcrops_cfg(padding=(0,0), resize=(0,0))
        
    def imcrops_cfg(self, padding=(0,0), resize=(0,0)):
        crops = []
        for box in self.boxes:
            # Apply padding
            box[0] -= padding[0]
            box[1] -= padding[1]
            box[2] += padding[0]
            box[3] += padding[1]

            # Ensure the coordinates are within bounds
            box[0] = max(0, box[0])  # Ensure x1 is not negative
            box[1] = max(0, box[1])  # Ensure y1 is not negative
            box[2] = min(self.imshape[1], box[2])  # Ensure x2 is within image width
            box[3] = min(self.imshape[0], box[3])  # Ensure y2 is within image height

            # Crop the image based on the adjusted box
            cropped_img = self.frame1[box[1]:box[3], box[0]:box[2]]

            # Resize cropped image to the specified dimensions if needed
            if resize:
                cropped_img = cv2.resize(cropped_img, resize)

            crops.append(cropped_img)
        return crops 

    def plot(self, image=None):
        if image is None:
            plot = self.frame1.copy()
        elif image.shape[:2] == self.frame1.shape[:2]:
            plot = image.copy()
        else:
            plot = cv2.resize(image, (self.frame1.shape[1], self.frame1.shape[0]))
        for box in self.boxes:
            cv2.rectangle(plot, box[:2], box[2:], (0,165,255), 2)
        return plot

class BaseMotionDetector(ABC):
    def __init__(self,
                 image_preprocess_func=None,
                 image_postprocess_func=None):
        self.image_preprocess_func = image_preprocess_func
        self.image_postprocess_func = image_postprocess_func

        self.frame1 = None
        self.frame2 = None

    @property
    @abstractmethod
    def boxes(self):
        pass
    
    @property
    @abstractmethod
    def immask(self):
        pass
    
    def __call__(self, frame1, frame2):
        if self.image_preprocess_func is None:
            self.frame1 = frame1.copy()
            self.frame2 = frame2.copy()
        else:
            self.frame1 = self.image_preprocess_func(frame1)
            self.frame2 = self.image_preprocess_func(frame2)
        return _GetResults(self)
    
    def __getattribute__(self, name):
        if name == 'immask':
            post_proc_func = object.__getattribute__(self, 'image_postprocess_func')
            mask = object.__getattribute__(self, name)
            if post_proc_func is not None and mask is not None:
                return post_proc_func(mask)
        return object.__getattribute__(self, name)
    
    # @property
    # def _conv_ignore_areas(self):
    #     if self.ignore_areas is not None and self.imshape() is not None:
    #         if np.max(self.ignore_areas) <= 1:
    #             h, w, ch = self.imshape()
    #             self.ignore_areas[:, :, 0] = self.ignore_areas[:, :, 0] * w
    #             self.ignore_areas[:, :, 1] = self.ignore_areas[:, :, 1] * h
    #     return self.ignore_areas
    
    # @property
    # def _conv_interested_areas(self):
    #     if self.interested_areas is not None and self.imshape() is not None:
    #         if np.max(self.interested_areas) <= 1:
    #             h, w, ch = self.imshape()
    #             self.interested_areas[:, :, 0] = self.interested_areas[:, :, 0] * w
    #             self.interested_areas[:, :, 1] = self.interested_areas[:, :, 1] * h
    #     return self.interested_areas

    # def set_ignore_areas(self, ignore_areas):
    #     self.ignore_areas = ignore_areas

    # def set_interested_areas(self, interested_areas):
    #     self.interested_areas = interested_areas
    
    # def __getattribute__(self, name):
    #     if name == 'boxes':
    #         _boxes = ignore_areas(
    #             boxes=object.__getattribute__(self, name),
    #             areas=object.__getattribute__(self, '_conv_ignore_areas')
    #         )
    #         _boxes = interested_areas(
    #             _boxes,
    #             areas=object.__getattribute__(self, '_conv_interested_areas')
    #         )
    #         return _boxes
        
    #     elif name == 'ignore_areas':
    #         if 'ignore_areas' in object.__dir__(self):
    #             return object.__getattribute__(self, 'ignore_areas')
    #         else:
    #             return None
            
    #     elif name == 'interested_areas':
    #         if 'interested_areas' in object.__dir__(self):
    #             return object.__getattribute__(self, 'interested_areas')
    #         else:
    #             return None
    #     return super().__getattribute__(name)
