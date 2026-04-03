import os
import cv2
import yaml
from glob import glob
from mask_extraction_yolo import MaskExtractorYOLO
import sys
# Add parent dir to path to import common
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.visualization import save_visualization

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configurations
    common_cfg = load_config('configs/common.yaml')
    baseline_cfg = load_config('configs/part1_baseline.yaml')

    datasets_dir = common_cfg['datasets']['bmx_trees']
    output_viz_dir = common_cfg['output_dir']['visualizations']
    os.makedirs(output_viz_dir, exist_ok=True)

    # Initialize YOLO Mask Extractor
    extractor = MaskExtractorYOLO(
        model_path=baseline_cfg['model']['yolo_seg_path'],
        target_classes=baseline_cfg['extraction']['target_classes']
    )

    # Read video frames
    image_paths = sorted(glob(os.path.join(datasets_dir, '*.jpg')))
    if not image_paths:
        print(f"No images found in {datasets_dir}!")
        return

    print(f"Found {len(image_paths)} frames. Starting processing...")

    for i, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        # Step 2: Semantic Extraction
        masks, bboxes, class_ids = extractor.extract(frame)
        
        # Save visualization for verification
        out_path = os.path.join(output_viz_dir, f"frame_{i:04d}.jpg")
        save_visualization(frame, masks, bboxes, class_ids, out_path)
        
        print(f"Processed frame {i+1}/{len(image_paths)}")

if __name__ == '__main__':
    main()
