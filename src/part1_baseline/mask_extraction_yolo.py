import numpy as np
from ultralytics import YOLO

class MaskExtractorYOLO:
    def __init__(self, model_path, target_classes=None):
        """
        Initialize the YOLO segmentation model.
        :param model_path: Path to the YOLOv8-seg model weights.
        :param target_classes: List of class IDs to filter (e.g., [0, 1] for Person, Bicycle).
        """
        # Download or load the model
        self.model = YOLO(model_path)
        self.target_classes = target_classes

    def extract(self, frame):
        """
        Perform inference and extract masks for target classes.
        :param frame: BGR image frame from OpenCV.
        :return: Tuple of (masks, bboxes, class_ids)
                 masks: list of binary numpy arrays (H, W) or empty list
        """
        # Perform inference
        # Classes can be filtered directly in YOLO argument, or manually.
        results = self.model(frame, classes=self.target_classes, verbose=False)
        
        valid_masks = []
        valid_bboxes = []
        valid_class_ids = []
        
        if len(results) > 0 and results[0].masks is not None:
            # results[0].masks.data contains masks of shape (N, H, W)
            # Resize masks back to original frame size if needed (ultralytics does this automatically depending on usage, but masks.data might be resized)
            # A safer way to get original size masks in ultralytics is masks.xy or masks.data
            result = results[0]
            for i, mask in enumerate(result.masks.data):
                # Class check (already filtered by YOLO, but double check)
                cls_id = int(result.boxes.cls[i].item())
                if self.target_classes is None or cls_id in self.target_classes:
                    # Convert tensor mask to numpy array (H, W), resize to match frame shape
                    mask_np = mask.cpu().numpy()
                    
                    # Usually YOLO masks are a specific shape, might need resizing to match frame
                    h, w = frame.shape[:2]
                    if mask_np.shape != (h, w):
                        import cv2
                        mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Convert to strictly boolean/uint8 0 or 255
                    mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                    
                    valid_masks.append(mask_binary)
                    valid_bboxes.append(result.boxes.xyxy[i].cpu().numpy())
                    valid_class_ids.append(cls_id)
                    
        return valid_masks, valid_bboxes, valid_class_ids
