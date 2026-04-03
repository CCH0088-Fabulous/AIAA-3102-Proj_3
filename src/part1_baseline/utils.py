import os

import cv2
import numpy as np


def get_sequence_name(sequence_dir):
    return os.path.basename(os.path.normpath(sequence_dir))


def build_mask_output_path(mask_root, sequence_name):
    mask_dir = os.path.join(mask_root, sequence_name)
    os.makedirs(mask_dir, exist_ok=True)
    return mask_dir


def merge_instance_masks(masks, frame_shape):
    merged_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    for mask in masks:
        merged_mask = np.maximum(merged_mask, mask.astype(np.uint8))
    return merged_mask


def save_mask(mask, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask)
