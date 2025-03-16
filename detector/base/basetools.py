from typing import List
from shapely.geometry import Polygon, box as shapely_Box
import numpy as np
import math
from scipy.spatial.distance import cdist

def ignore_areas(boxes: List[List], areas: List[Polygon]):
    if areas is None:
        return boxes

    _boxes = []
    
    for box in boxes:
        polybox = shapely_Box(*box)
        for area in areas:
            polygon = Polygon(area)
            if not polybox.intersects(polygon):
                _boxes.append(box)

    return np.array(_boxes, dtype=np.int32)

def interested_areas(boxes: List[List], areas: List[Polygon]):
    if areas is None:
        return boxes
    
    _boxes = []

    for box in boxes:
        polybox = shapely_Box(*box)
        for area in areas:
            polygon = Polygon(area)
            if polybox.intersects(polygon):
                _boxes.append(box)

    return np.array(_boxes, dtype=np.int32)

def mixing_boxes(boxes: List[List], dist_thresh: int = 50):
    if len(boxes) <= 1:
        return boxes
    
    merged_boxes = []
    non_merged_boxes = []  # To track boxes that are not merged
    ii1 = []

    merged_indices = set()  # To track indices of merged boxes

    for i1, box1 in enumerate(boxes):
        ii2 = []
        for i2, box2 in enumerate(boxes):
            if i1 != i2:
                bbox1 = np.array(box1).reshape(-1, 2)
                bbox2 = np.array(box2).reshape(-1, 2)

                # Calculate the distance between the bounding boxes
                axisx = max(bbox1[:, 0]) - min(bbox2[:, 0])
                axisy = max(bbox1[:, 1]) - min(bbox2[:, 1])
                dist = math.sqrt(axisx**2 + axisy**2)

                # If boxes are close enough, merge them
                if dist < dist_thresh:
                    ii2.append(i2)
                    merged_indices.add(i1)  # Mark i1 as merged
                    merged_indices.add(i2)  # Mark i2 as merged
                    
                    # Calculate the new bounding box (union of both boxes)
                    new_box = [
                        min(bbox1[:, 0].min(), bbox2[:, 0].min()),  # min x
                        min(bbox1[:, 1].min(), bbox2[:, 1].min()),  # min y
                        max(bbox1[:, 0].max(), bbox2[:, 0].max()),  # max x
                        max(bbox1[:, 1].max(), bbox2[:, 1].max())   # max y
                    ]
                    merged_boxes.append(new_box)
        ii1.append(ii2)

    # Remove duplicates in merged_boxes
    merged_boxes = [list(x) for x in set(tuple(x) for x in merged_boxes)]

    # Now, find the non-merged boxes (those not in merged_indices)
    for i, box in enumerate(boxes):
        if i not in merged_indices:
            non_merged_boxes.append(box)

    merged_boxes = np.array(merged_boxes).reshape(-1,4).astype(np.int32)
    non_merged_boxes = np.array(non_merged_boxes).reshape(-1, 4).astype(np.int32)

    return np.concatenate([merged_boxes, non_merged_boxes],axis=0)


def combine_boxes(bboxes, distance_thres=50):
    """
    Combine bounding boxes that overlap or are within a certain distance threshold
    based on the nearest sides of the boxes using a proper distance function.

    Parameters:
    - bboxes (np.array): Array of bounding boxes with shape (n_box, 4), where each row is [x1, y1, x2, y2].
    - distance_thres (int): Distance threshold in pixels to consider for combining bounding boxes.

    Returns:
    - combined_bboxes (np.array): Array of combined bounding boxes with shape (m_box, 4).
    """
    if len(bboxes) <= 1:
        return bboxes

    def is_overlap_or_near(bbox1, bbox2, distance_thres):
        """
        Check if two bounding boxes overlap or are within a distance threshold
        based on the nearest sides of the boxes using a proper distance function.
        """
        # Check if the bounding boxes overlap
        x1_overlap = max(bbox1[0], bbox2[0])
        y1_overlap = max(bbox1[1], bbox2[1])
        x2_overlap = min(bbox1[2], bbox2[2])
        y2_overlap = min(bbox1[3], bbox2[3])

        overlap_area = max(0, x2_overlap - x1_overlap) * max(0, y2_overlap - y1_overlap)

        if overlap_area > 0:
            return True  # Overlapping

        # Extract the coordinates of the edges of the bounding boxes
        bbox1_edges = np.array([
            [bbox1[0], bbox1[1]],  # Top-left
            [bbox1[2], bbox1[1]],  # Top-right
            [bbox1[0], bbox1[3]],  # Bottom-left
            [bbox1[2], bbox1[3]]   # Bottom-right
        ])

        bbox2_edges = np.array([
            [bbox2[0], bbox2[1]],  # Top-left
            [bbox2[2], bbox2[1]],  # Top-right
            [bbox2[0], bbox2[3]],  # Bottom-left
            [bbox2[2], bbox2[3]]   # Bottom-right
        ])

        # Calculate the pairwise distances between all edges of the two bounding boxes
        distances = cdist(bbox1_edges, bbox2_edges)

        # Find the minimum distance between the edges
        nearest_distance = np.min(distances)

        return nearest_distance <= distance_thres

    def combine_two_bboxes(bbox1, bbox2):
        """
        Combine two bounding boxes into one.
        """
        x1 = min(bbox1[0], bbox2[0])
        y1 = min(bbox1[1], bbox2[1])
        x2 = max(bbox1[2], bbox2[2])
        y2 = max(bbox1[3], bbox2[3])
        return [x1, y1, x2, y2]

    # Sort by xmin
    bboxes = bboxes[bboxes[:, 0].argsort()] 

    # Initialize a list to keep track of which bounding boxes have been merged
    merged = [False] * len(bboxes)
    combined_bboxes = []

    for i in range(len(bboxes)):
        if merged[i]:
            continue  # Skip if already merged

        current_bbox = bboxes[i]
        for j in range(i + 1, len(bboxes)):
            if merged[j]:
                continue  # Skip if already merged

            if is_overlap_or_near(current_bbox, bboxes[j], distance_thres):
                # Combine the bounding boxes
                current_bbox = combine_two_bboxes(current_bbox, bboxes[j])
                merged[j] = True  # Mark as merged

        combined_bboxes.append(current_bbox)
        merged[i] = True  # Mark as merged

    return np.array(combined_bboxes)