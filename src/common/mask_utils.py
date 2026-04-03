import cv2
import numpy as np


def normalize_binary_mask(mask):
	if mask is None:
		return None
	return ((mask > 0).astype(np.uint8)) * 255


def dilate_mask(mask, kernel_size=5, iterations=1):
	binary_mask = normalize_binary_mask(mask)
	if binary_mask is None:
		return None
	if kernel_size <= 1 or iterations <= 0:
		return binary_mask

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
	return cv2.dilate(binary_mask, kernel, iterations=iterations)


def fill_mask_holes(mask):
	binary_mask = normalize_binary_mask(mask)
	if binary_mask is None:
		return None

	height, width = binary_mask.shape[:2]
	padded_mask = cv2.copyMakeBorder(binary_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
	flood_fill_mask = np.zeros((height + 4, width + 4), dtype=np.uint8)
	filled = padded_mask.copy()
	cv2.floodFill(filled, flood_fill_mask, (0, 0), 255)
	holes = cv2.bitwise_not(filled)[1:-1, 1:-1]
	return cv2.bitwise_or(binary_mask, holes)


def remove_small_connected_components(mask, min_area=0):
	binary_mask = normalize_binary_mask(mask)
	if binary_mask is None or min_area <= 0:
		return binary_mask

	num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
	cleaned_mask = np.zeros_like(binary_mask)
	for label_index in range(1, num_labels):
		component_area = stats[label_index, cv2.CC_STAT_AREA]
		if component_area >= min_area:
			cleaned_mask[labels == label_index] = 255
	return cleaned_mask


def temporal_smooth_mask(current_mask, previous_masks=None, temporal_window=1, min_votes=None):
	binary_mask = normalize_binary_mask(current_mask)
	if binary_mask is None:
		return None
	if temporal_window <= 1:
		return binary_mask

	mask_stack = []
	if previous_masks:
		mask_stack.extend(normalize_binary_mask(mask) for mask in previous_masks if mask is not None)
	mask_stack.append(binary_mask)

	if len(mask_stack) == 1:
		return binary_mask

	mask_stack = mask_stack[-temporal_window:]
	stacked = np.stack([(mask > 0).astype(np.uint8) for mask in mask_stack], axis=0)
	required_votes = min_votes if min_votes is not None else (stacked.shape[0] // 2 + 1)
	smoothed = (np.sum(stacked, axis=0) >= required_votes).astype(np.uint8) * 255
	return smoothed


def postprocess_mask(mask, postprocess_cfg=None, previous_masks=None):
	cfg = postprocess_cfg or {}

	processed_mask = normalize_binary_mask(mask)
	if processed_mask is None:
		return None

	processed_mask = dilate_mask(
		processed_mask,
		kernel_size=int(cfg.get("dilate_kernel_size", 0)),
		iterations=int(cfg.get("dilate_iterations", 1)),
	)

	if cfg.get("fill_holes", False):
		processed_mask = fill_mask_holes(processed_mask)

	processed_mask = remove_small_connected_components(
		processed_mask,
		min_area=int(cfg.get("min_component_area", 0)),
	)

	processed_mask = temporal_smooth_mask(
		processed_mask,
		previous_masks=previous_masks,
		temporal_window=int(cfg.get("temporal_window", 1)),
		min_votes=cfg.get("temporal_min_votes"),
	)
	return normalize_binary_mask(processed_mask)
