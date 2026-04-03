import cv2
import numpy as np

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
	sys.path.append(str(SRC_DIR))

from common.mask_utils import normalize_binary_mask


class TraditionalVideoInpainter:
	"""Baseline video inpainter with temporal borrowing and spatial fallback."""

	def __init__(self, temporal_window=3, spatial_fallback="telea", radius=3.0):
		self.temporal_window = max(int(temporal_window), 0)
		self.spatial_fallback = spatial_fallback.lower()
		self.radius = float(radius)
		self.inpaint_flag = self._resolve_inpaint_flag(self.spatial_fallback)

	def _resolve_inpaint_flag(self, fallback_name):
		if fallback_name in {"navier-stokes", "navier_stokes", "ns"}:
			return cv2.INPAINT_NS
		return cv2.INPAINT_TELEA

	def _candidate_indices(self, frame_index, sequence_length):
		for offset in range(1, self.temporal_window + 1):
			previous_index = frame_index - offset
			next_index = frame_index + offset
			if previous_index >= 0:
				yield previous_index
			if next_index < sequence_length:
				yield next_index

	def temporal_fill_frame(self, frames, masks, frame_index):
		source_frame = frames[frame_index]
		source_mask = normalize_binary_mask(masks[frame_index])
		if source_mask is None or cv2.countNonZero(source_mask) == 0:
			return source_frame.copy(), np.zeros(source_frame.shape[:2], dtype=np.uint8), {
				"temporal_filled_pixels": 0,
				"fallback_pixels": 0,
				"remaining_pixels": 0,
				"borrowed_frames": 0,
			}

		restored_frame = source_frame.copy()
		remaining_mask = source_mask.copy()
		borrowed_frames = 0
		temporal_filled_pixels = 0

		for candidate_index in self._candidate_indices(frame_index, len(frames)):
			candidate_mask = normalize_binary_mask(masks[candidate_index])
			if candidate_mask is None:
				continue

			available_background = (remaining_mask > 0) & (candidate_mask == 0)
			if not np.any(available_background):
				continue

			restored_frame[available_background] = frames[candidate_index][available_background]
			filled_count = int(np.count_nonzero(available_background))
			temporal_filled_pixels += filled_count
			remaining_mask[available_background] = 0
			borrowed_frames += 1

			if cv2.countNonZero(remaining_mask) == 0:
				break

		return restored_frame, remaining_mask, {
			"temporal_filled_pixels": temporal_filled_pixels,
			"fallback_pixels": 0,
			"remaining_pixels": int(cv2.countNonZero(remaining_mask)),
			"borrowed_frames": borrowed_frames,
		}

	def inpaint_frame(self, frames, masks, frame_index):
		restored_frame, remaining_mask, stats = self.temporal_fill_frame(frames, masks, frame_index)

		remaining_pixels = int(cv2.countNonZero(remaining_mask))
		if remaining_pixels > 0:
			restored_frame = cv2.inpaint(
				restored_frame,
				remaining_mask,
				self.radius,
				self.inpaint_flag,
			)
			stats["fallback_pixels"] = remaining_pixels
			stats["remaining_pixels"] = 0

		return restored_frame, stats

	def inpaint_sequence(self, frames, masks):
		if len(frames) != len(masks):
			raise ValueError("Frames and masks must have the same length for sequence inpainting.")

		restored_frames = []
		per_frame_stats = []
		for frame_index in range(len(frames)):
			restored_frame, stats = self.inpaint_frame(frames, masks, frame_index)
			restored_frames.append(restored_frame)
			per_frame_stats.append(stats)

		return restored_frames, per_frame_stats
