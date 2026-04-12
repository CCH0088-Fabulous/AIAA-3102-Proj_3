import os

import cv2
import numpy as np
import imageio

def get_sequence_name(sequence_dir):
    return os.path.basename(os.path.normpath(sequence_dir))


def build_mask_output_path(mask_root, sequence_name):
    mask_dir = os.path.join(mask_root, sequence_name)
    os.makedirs(mask_dir, exist_ok=True)
    return mask_dir


def build_video_output_path(video_root, video_filename):
    os.makedirs(video_root, exist_ok=True)
    return os.path.join(video_root, video_filename)


def merge_instance_masks(masks, frame_shape):
    merged_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    for mask in masks:
        merged_mask = np.maximum(merged_mask, mask.astype(np.uint8))
    return merged_mask


def save_mask(mask, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask)


def write_video(frames, output_path, fps=30, codec="mp4v"):
    if not frames:
        raise ValueError("Cannot write a video without frames.")

    height, width = frames[0].shape[:2]
    
    rgb_frames = []
    for frame in frames:
        if frame.shape[:2] != (height, width):
            # width is dsize[0], height is dsize[1]
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # imageio uses ffmpeg natively to write compliant standard mp4 files
    imageio.mimwrite(output_path, rgb_frames, fps=fps, macro_block_size=None)
