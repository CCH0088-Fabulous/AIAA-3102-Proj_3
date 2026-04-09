import os

import cv2
import numpy as np

from common.mask_utils import normalize_binary_mask


DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_MASK_COLORS = [
	(0, 200, 0),
	(0, 180, 255),
	(255, 180, 0),
	(255, 80, 80),
	(180, 0, 255),
	(80, 255, 255),
]


def ensure_visualization_dirs(root_dir, sequence_name):
	base_dir = os.path.join(root_dir, sequence_name)
	subdirs = {
		"base": base_dir,
		"motion_scores": os.path.join(base_dir, "motion_scores"),
		"mask_overlays": os.path.join(base_dir, "mask_overlays"),
		"comparisons": os.path.join(base_dir, "comparisons"),
	}
	for directory in subdirs.values():
		os.makedirs(directory, exist_ok=True)
	return subdirs


def save_visualization_frame(image, output_path):
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	cv2.imwrite(output_path, image)


def _blend_mask(frame, mask, color, alpha):
	binary_mask = normalize_binary_mask(mask)
	blended = frame.copy()
	if binary_mask is None or cv2.countNonZero(binary_mask) == 0:
		return blended

	color_layer = np.zeros_like(frame)
	color_layer[:] = np.array(color, dtype=np.uint8)
	mask_region = binary_mask > 0
	blended[mask_region] = cv2.addWeighted(
		frame[mask_region],
		1.0 - alpha,
		color_layer[mask_region],
		alpha,
		0,
	)
	return blended


def _mask_anchor(mask):
	binary_mask = normalize_binary_mask(mask)
	if binary_mask is None or cv2.countNonZero(binary_mask) == 0:
		return (8, 24)

	ys, xs = np.where(binary_mask > 0)
	top = int(np.min(ys))
	left = int(np.min(xs))
	return (left, max(top - 8, 18))


def _draw_mask_contours(frame, mask, color, thickness=2):
	binary_mask = normalize_binary_mask(mask)
	if binary_mask is None or cv2.countNonZero(binary_mask) == 0:
		return frame

	contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(frame, contours, -1, color, thickness)
	return frame


def _draw_label(frame, text, anchor, color):
	x, y = anchor
	(width, height), baseline = cv2.getTextSize(text, DEFAULT_FONT, 0.45, 1)
	top_left = (x, max(y - height - baseline - 4, 0))
	bottom_right = (x + width + 6, y + 4)
	cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), thickness=-1)
	cv2.putText(frame, text, (x + 3, y - 2), DEFAULT_FONT, 0.45, color, 1, cv2.LINE_AA)
	return frame


def _sample_flow_vectors(points_previous, points_current, max_vectors):
	if points_previous is None or points_current is None:
		return []
	if len(points_previous) == 0 or len(points_current) == 0:
		return []

	total = min(len(points_previous), len(points_current))
	if total <= max_vectors:
		indices = np.arange(total)
	else:
		indices = np.linspace(0, total - 1, max_vectors, dtype=int)
	return [(points_previous[index], points_current[index]) for index in indices]


def render_motion_score_overlay(
	frame,
	candidate_masks,
	motion_summaries,
	overlay_alpha=0.35,
	draw_flow_vectors=True,
	max_flow_vectors=20,
):
	output = frame.copy()
	if not candidate_masks:
		return _draw_label(output, "No candidate instances", (8, 24), (255, 255, 255))

	for instance_index, mask in enumerate(candidate_masks):
		summary = motion_summaries[instance_index] if instance_index < len(motion_summaries) else {}
		if summary.get("selected") and summary.get("valid"):
			color = (0, 200, 0)
		elif summary.get("selected"):
			color = (0, 180, 255)
		else:
			color = (0, 0, 255)

		output = _blend_mask(output, mask, color, overlay_alpha)
		output = _draw_mask_contours(output, mask, color)

		score = summary.get("score")
		if score is None:
			score_text = "score=n/a"
		else:
			score_text = f"score={score:.2f}"

		tracked = summary.get("num_tracked", 0)
		status_text = "dynamic" if summary.get("selected") else "static"
		label = f"id={instance_index} {score_text} pts={tracked} {status_text}"
		output = _draw_label(output, label, _mask_anchor(mask), color)

		if draw_flow_vectors and summary.get("valid"):
			vector_pairs = _sample_flow_vectors(
				summary.get("points_previous"),
				summary.get("points_current"),
				max_flow_vectors,
			)
			for point_previous, point_current in vector_pairs:
				start = tuple(np.round(point_previous).astype(int))
				end = tuple(np.round(point_current).astype(int))
				cv2.arrowedLine(output, start, end, color, 1, tipLength=0.25)

	output = _draw_label(output, "Motion scoring", (8, 24), (255, 255, 255))
	return output


def render_instance_mask_overlay(
	frame,
	instance_masks,
	instance_labels=None,
	overlay_alpha=0.35,
	title="Tracked objects",
):
	output = frame.copy()
	if not instance_masks:
		output = _draw_label(output, "No tracked objects", (8, 52), (255, 255, 255))
		return _draw_label(output, title, (8, 24), (255, 255, 255))

	labels = instance_labels or []
	for index, mask in enumerate(instance_masks):
		color = DEFAULT_MASK_COLORS[index % len(DEFAULT_MASK_COLORS)]
		label = labels[index] if index < len(labels) else f"obj={index + 1}"
		output = _blend_mask(output, mask, color, overlay_alpha)
		output = _draw_mask_contours(output, mask, color)
		output = _draw_label(output, label, _mask_anchor(mask), color)

	return _draw_label(output, title, (8, 24), (255, 255, 255))


def render_mask_overlay(frame, mask, overlay_alpha=0.35, color=(255, 180, 0)):
	output = _blend_mask(frame.copy(), mask, color, overlay_alpha)
	output = _draw_mask_contours(output, mask, color)
	return _draw_label(output, "Removal mask", (8, 24), color)


def _pad_to_height(image, target_height):
	if image.shape[0] == target_height:
		return image
	if image.shape[0] > target_height:
		return cv2.resize(image, (image.shape[1], target_height), interpolation=cv2.INTER_LINEAR)

	pad_height = target_height - image.shape[0]
	return cv2.copyMakeBorder(image, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _annotate_panel(image, title):
	annotated = image.copy()
	bar_height = 28
	cv2.rectangle(annotated, (0, 0), (annotated.shape[1], bar_height), (0, 0, 0), thickness=-1)
	cv2.putText(annotated, title, (10, 19), DEFAULT_FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
	return annotated


def render_before_after_comparison(original_frame, mask_overlay_frame, restored_frame):
	original_panel = _annotate_panel(original_frame, "Original")
	mask_panel = _annotate_panel(mask_overlay_frame, "Mask Overlay")
	restored_panel = _annotate_panel(restored_frame, "Restored")

	target_height = max(original_panel.shape[0], mask_panel.shape[0], restored_panel.shape[0])
	original_panel = _pad_to_height(original_panel, target_height)
	mask_panel = _pad_to_height(mask_panel, target_height)
	restored_panel = _pad_to_height(restored_panel, target_height)
	return np.concatenate([original_panel, mask_panel, restored_panel], axis=1)

