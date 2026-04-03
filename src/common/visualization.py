import cv2
import numpy as np
import os

def save_visualization(frame, masks, bboxes, class_ids, output_path):
    """
    Overlay bounding boxes and masks on the original frame and save to path.
    :param frame: Base RGB/BGR image (HxWx3).
    :param masks: List of masks (HxWx1 or HxW numpy arrays, purely binary or 0-255).
    :param bboxes: List of bounding boxes [x1, y1, x2, y2].
    :param class_ids: List of class IDs for each mask.
    :param output_path: Where to save the output image.
    """
    viz = frame.copy()
    
    # Simple color map
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # G, R, B
    
    for i, mask in enumerate(masks):
        color = colors[class_ids[i] % len(colors)]
        
        # Overlay mask using low opacity
        colored_mask = np.zeros_like(viz)
        if mask.max() == 1:
            mask = mask * 255
            
        colored_mask[mask > 0] = color
        alpha = 0.5
        cv2.addWeighted(colored_mask, alpha, viz, 1 - alpha, 0, viz)
        
        # Draw bounding boxes (Optional)
        if bboxes is not None and len(bboxes) > 0:
            box = bboxes[i]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
            cv2.putText(viz, f'Class: {class_ids[i]}', (x1, max(y1-10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Ensure output directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, viz)
