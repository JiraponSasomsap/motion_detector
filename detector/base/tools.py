import cv2
import numpy as np

def get_mask(frame1, frame2, mask_kernel_size=(9,9)):
    """Generate a binary mask of moving areas using frame differencing.
    
    Args:
        frame1 (numpy.ndarray): First frame (previous frame) in BGR format.
        frame2 (numpy.ndarray): Second frame (current frame) in BGR format.
        mask_kernel_size (tuple, optional): Size of the morphological kernel. Default is (9,9).
    
    Returns:
        numpy.ndarray: A binary mask where moving areas are white (255) and static areas are black (0).
    """
    # Convert both frames to grayscale for easier comparison
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Create a morphological kernel for post-processing the mask
    mask_kernel = np.ones(mask_kernel_size, dtype=np.uint8)

    # Compute absolute difference between two frames to detect motion
    diff = cv2.absdiff(frame1_gray, frame2_gray)
    
    # Apply thresholding to generate a binary mask
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Perform morphological closing to fill small gaps in the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mask_kernel)
    
    return mask

def get_contour_detections(mask, bbox_thresh):
    """Extract bounding boxes from contours in the mask.

    Args:
        mask (numpy.ndarray): Binary mask (0 and 255) from which contours are extracted.
        bbox_thresh (int): Minimum area threshold to filter small detections.

    Returns:
        numpy.ndarray: Array of detected bounding boxes in the format [x_min, y_min, x_max, y_max, area].
    """
    # Find external contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # Get bounding box
        area = w * h  # Compute area of the bounding box

        # Filter out small detections
        if area > bbox_thresh:
            detections.append([x, y, x + w, y + h, area])

    return np.array(detections) if detections else np.empty((0, 5), dtype=np.int32)

def compute_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (tuple or list): Bounding box in the format (x1, y1, x2, y2).
        box2 (tuple or list): Bounding box in the format (x1, y1, x2, y2).

    Returns:
        float: IoU value between 0.0 and 1.0.
    """
    x11, y11, x21, y21 = box1
    x12, y12, x22, y22 = box2

    # Compute intersection coordinates
    xi1 = max(x11, x12)
    yi1 = max(y11, y12)
    xi2 = min(x21, x22)
    yi2 = min(y21, y22)

    # Compute intersection area (ensure it's non-negative)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0  # No overlap

    # Compute individual box areas
    box1_area = (x21 - x11) * (y21 - y11)
    box2_area = (x22 - x12) * (y22 - y12)

    # Compute IoU
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def non_max_suppression(boxes, scores, nms_thresh):
    """Apply Non-Maximum Suppression (NMS) to remove redundant bounding boxes.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes in the format [[x1, y1, x2, y2], ...].
        scores (numpy.ndarray): Confidence scores associated with each bounding box.
        nms_thresh (float): IoU threshold for suppressing overlapping boxes.

    Returns:
        numpy.ndarray: Array of selected bounding boxes after applying NMS.
    """
    # Sort boxes by scores in descending order
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]  # Reorder boxes based on scores

    keep = []
    while len(boxes) > 0:
        current_box = boxes[0]
        keep.append(current_box)

        # Compute IoU for remaining boxes and filter out overlapping ones
        boxes = np.array([
            box for box in boxes[1:]
            if compute_iou(current_box, box) < nms_thresh
        ])

    return np.array(keep) if keep else np.empty((0, 4), dtype=np.float32)

def compute_flow(frame1, frame2):
    """
    Computes optical flow between two consecutive frames using the Farneback method.

    Parameters:
        frame1 (np.ndarray):(First frame, BGR image)
        frame2 (np.ndarray):(Second frame, BGR image)

    Returns:
        flow (np.ndarray):(Optical flow vectors)
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur (adjust parameters if needed)
    gray1 = cv2.GaussianBlur(gray1, (3,3), sigmaX=1.5)
    gray2 = cv2.GaussianBlur(gray2, (3,3), sigmaX=1.5)

    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                        pyr_scale=0.5,  # Reduced for better accuracy
                                        levels=3,
                                        winsize=15,  # Increased for robustness
                                        iterations=5,  # Increased for stability
                                        poly_n=7,  # Reduced for smoother results
                                        poly_sigma=1.5,
                                        flags=0)
    return flow

def get_motion_mask(flow_mag, motion_thresh, mask_kernel):
    """
    Generate a binary mask highlighting motion regions based on optical flow magnitude.

    Args:
        flow_mag (np.ndarray): Optical flow magnitude matrix.
        motion_thresh (float): Threshold value to classify motion pixels.
        mask_kernel (np.ndarray): Structuring element for morphological operations.

    Returns:
        np.ndarray: Binary motion mask (same size as flow_mag, with 255 for motion pixels and 0 otherwise).
    """

    # Convert flow magnitude to binary motion mask
    motion_mask = (flow_mag > motion_thresh).astype(np.uint8) * 255

    # Morphological operations to refine the mask
    motion_mask = cv2.erode(motion_mask, mask_kernel, iterations=1)  # Reduce noise
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, mask_kernel, iterations=1)  # Remove small noise
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, mask_kernel, iterations=3)  # Fill small holes

    return motion_mask

def get_contour_detections_angle(mask, ang, angle_thresh=1, thresh=100):
    """
    Extracts bounding boxes from contours in a mask while considering flow angles.

    Args:
        mask (np.ndarray): Binary mask (thresholded image).
        ang (np.ndarray): Flow angle matrix.
        angle_thresh (float): Multiplication factor for angle standard deviation filtering.
        thresh (int): Minimum area threshold for valid contours.

    Returns:
        np.ndarray: Array of bounding boxes and scores in the format [[x1, y1, x2, y2, score]].
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    detections = []
    
    # Compute global angle threshold
    global_angle_thresh = angle_thresh * ang.std()

    for cnt in contours:
        # Create a new mask for each contour
        temp_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(temp_mask, [cnt], 0, (255,), -1)

        # Get bounding box and area
        x, y, w, h = cv2.boundingRect(cnt)
        area = int(w) * int(h)

        # Extract flow angles within the contour region
        flow_angle = np.where(temp_mask == 255, ang, np.nan).flatten()
        flow_angle = flow_angle[~np.isnan(flow_angle)]  # Remove NaNs

        # Filter based on area and angle standard deviation
        if area > thresh and np.std(flow_angle) < global_angle_thresh:
            detections.append([x, y, x + w, y + h, area])

    return np.array(detections, dtype=np.float32)

def get_motion_mask_morphologyEx(fg_mask, min_thresh=25, kernel=None):
    """Obtains a thresholded motion mask using morphological operations.
    
    Parameters:
        fg_mask (np.ndarray): Foreground mask (grayscale or color).
        min_thresh (int): Threshold for motion detection.
        kernel (np.ndarray, optional): Morphological kernel. Defaults to 9x9.
    
    Returns:
        np.ndarray: Binary motion mask.
    """
    if fg_mask.ndim == 3:  
        fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)  
    
    if kernel is None:
        kernel = np.ones((9, 9), dtype=np.uint8)

    min_thresh = max(1, min_thresh)  # Ensure nonzero threshold
    _, thresh = cv2.threshold(fg_mask, min_thresh, 255, cv2.THRESH_BINARY)
    
    motion_mask = cv2.medianBlur(thresh, 3)
    
    # Morphological operations
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return motion_mask

def check_intersection(boxA, boxB):
    """
    Check if two bounding boxes intersect.
    :param boxA: (x1, y1, x2, y2)
    :param boxB: (x1, y1, x2, y2)
    :return: True if they intersect, False otherwise
    """
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    # Find intersection area
    x_left = max(x1A, x1B)
    y_top = max(y1A, y1B)
    x_right = min(x2A, x2B)
    y_bottom = min(y2A, y2B)

    return x_right > x_left and y_bottom > y_top  # True if intersection exists

def boxes_padding(boxes, padding):
    _boxes = boxes.copy()
    _boxes[0] -= padding[0]
    _boxes[1] -= padding[1]
    _boxes[2] += padding[0]
    _boxes[3] += padding[1]
    return _boxes

def merge_boxes(boxes, padding=(100, 200)):

    """
    Merge overlapping bounding boxes with padding.
    
    :param boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
    :param padding: Tuple (pad_x, pad_y) to expand boxes before merging
    :return: List of merged bounding boxes
    """

    if not boxes:
        return []

    clone_boxes = np.array(boxes, dtype=np.int32)
    merged_boxes = []

    while len(clone_boxes) > 0:
        base_box = clone_boxes[0].copy()
        base_box = boxes_padding(base_box, padding)

        to_remove = [0]  # Mark first box for removal
        for idx in range(1, len(clone_boxes)):
            expanded_box = clone_boxes[idx].copy()
            expanded_box = boxes_padding(expanded_box, padding)

            if check_intersection(base_box, expanded_box):
                # Merge the boxes
                base_box = [
                    min(clone_boxes[0][0], clone_boxes[idx][0]),  # min x1
                    min(clone_boxes[0][1], clone_boxes[idx][1]),  # min y1
                    max(clone_boxes[0][2], clone_boxes[idx][2]),  # max x2
                    max(clone_boxes[0][3], clone_boxes[idx][3])   # max y2
                ]
                base_box = boxes_padding(base_box, padding)

                to_remove.append(idx)

        merged_boxes.append(base_box)

        # Remove merged boxes
        clone_boxes = np.delete(clone_boxes, to_remove, axis=0)

    return merged_boxes
            

def get_bounding_boxes(image):
    """Returns bounding boxes (x1, y1, x2, y2) for contours in an image."""
    
    if len(image.shape) != 3:
        gray = image.copy()
    else:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding or edge detection to get a binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store bounding boxes
    bounding_boxes = []
    
    # Loop through each contour to find the bounding box
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_boxes.append((x, y, x + w, y + h))  # (x1, y1, x2, y2)
    
    return bounding_boxes

def is_daytime_histogram(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    elif len(image.shape) == 2:
        gray = image.copy()
    else:
        raise

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])  # Compute histogram

    bright_pixels = np.sum(hist[120:])  # Count bright pixels (above 150 intensity)
    dark_pixels = np.sum(hist[:100])  # Count dark pixels (below 50 intensity)

    return bright_pixels > dark_pixels