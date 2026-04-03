import cv2
import numpy as np


DEFAULT_FEATURE_PARAMS = {
	"max_corners": 200,
	"quality_level": 0.01,
	"min_distance": 5,
	"block_size": 7,
	"mask_dilate_kernel_size": 5,
}

DEFAULT_LK_PARAMS = {
	"win_size": 21,
	"max_level": 3,
	"criteria": (
		cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
		30,
		0.01,
	),
	"max_backward_error": 1.5,
}


def ensure_grayscale(frame):
	if frame is None:
		raise ValueError("Expected a valid frame, but received None.")
	if frame.ndim == 2:
		return frame
	return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def normalize_mask(mask):
	if mask is None:
		return None
	return ((mask > 0).astype(np.uint8)) * 255


def _build_feature_mask(mask, mask_dilate_kernel_size):
	feature_mask = normalize_mask(mask)
	if feature_mask is None:
		return None

	if mask_dilate_kernel_size and mask_dilate_kernel_size > 1:
		kernel = cv2.getStructuringElement(
			cv2.MORPH_ELLIPSE,
			(mask_dilate_kernel_size, mask_dilate_kernel_size),
		)
		feature_mask = cv2.dilate(feature_mask, kernel, iterations=1)

	return feature_mask


def extract_feature_points(gray_frame, mask, feature_params=None):
	params = dict(DEFAULT_FEATURE_PARAMS)
	if feature_params:
		params.update(feature_params)

	feature_mask = _build_feature_mask(mask, params.pop("mask_dilate_kernel_size", 0))
	if feature_mask is None or cv2.countNonZero(feature_mask) == 0:
		return None

	open_cv_params = {
		"maxCorners": params.pop("max_corners", 200),
		"qualityLevel": params.pop("quality_level", 0.01),
		"minDistance": params.pop("min_distance", 5),
		"blockSize": params.pop("block_size", 7),
	}
	open_cv_params.update(params)

	return cv2.goodFeaturesToTrack(gray_frame, mask=feature_mask, **open_cv_params)


def _prepare_lk_params(lk_params=None):
	params = dict(DEFAULT_LK_PARAMS)
	if lk_params:
		params.update(lk_params)

	win_size = params.pop("win_size", 21)
	if isinstance(win_size, int):
		params["winSize"] = (win_size, win_size)
	else:
		params["winSize"] = tuple(win_size)

	params["maxLevel"] = params.pop("max_level", 3)
	params.setdefault(
		"criteria",
		(
			cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
			30,
			0.01,
		),
	)
	max_backward_error = params.pop("max_backward_error", 1.5)
	return params, max_backward_error


def track_points_lk(prev_gray, curr_gray, points, lk_params=None):
	if points is None or len(points) == 0:
		empty_points = np.empty((0, 2), dtype=np.float32)
		empty_values = np.empty((0,), dtype=np.float32)
		return empty_points, empty_points, empty_values, empty_values

	lk_args, max_backward_error = _prepare_lk_params(lk_params)
	next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points, None, **lk_args)
	if next_points is None or status is None:
		empty_points = np.empty((0, 2), dtype=np.float32)
		empty_values = np.empty((0,), dtype=np.float32)
		return empty_points, empty_points, empty_values, empty_values

	status = status.reshape(-1).astype(bool)
	previous_points = points.reshape(-1, 2)
	next_points = next_points.reshape(-1, 2)

	if np.any(status):
		backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
			curr_gray,
			prev_gray,
			next_points.reshape(-1, 1, 2),
			None,
			**lk_args,
		)
		if backward_points is not None and backward_status is not None:
			backward_points = backward_points.reshape(-1, 2)
			backward_status = backward_status.reshape(-1).astype(bool)
			backward_error = np.linalg.norm(previous_points - backward_points, axis=1)
			status &= backward_status & (backward_error <= max_backward_error)
		else:
			status[:] = False

	valid_previous = previous_points[status]
	valid_next = next_points[status]
	displacements = valid_next - valid_previous
	magnitudes = np.linalg.norm(displacements, axis=1).astype(np.float32)
	return valid_previous, valid_next, displacements, magnitudes


def estimate_mask_motion(
	previous_frame,
	current_frame,
	mask,
	feature_params=None,
	lk_params=None,
	min_tracked_points=8,
):
	previous_gray = ensure_grayscale(previous_frame)
	current_gray = ensure_grayscale(current_frame)

	feature_points = extract_feature_points(previous_gray, mask, feature_params=feature_params)
	num_features = 0 if feature_points is None else len(feature_points)

	previous_points, current_points, displacements, magnitudes = track_points_lk(
		previous_gray,
		current_gray,
		feature_points,
		lk_params=lk_params,
	)

	if magnitudes.size == 0:
		return {
			"valid": False,
			"num_features": num_features,
			"num_tracked": 0,
			"mean_motion": 0.0,
			"median_motion": 0.0,
			"max_motion": 0.0,
			"displacements": displacements,
			"magnitudes": magnitudes,
			"points_previous": previous_points,
			"points_current": current_points,
		}

	return {
		"valid": magnitudes.size >= int(min_tracked_points),
		"num_features": num_features,
		"num_tracked": int(magnitudes.size),
		"mean_motion": float(np.mean(magnitudes)),
		"median_motion": float(np.median(magnitudes)),
		"max_motion": float(np.max(magnitudes)),
		"displacements": displacements,
		"magnitudes": magnitudes,
		"points_previous": previous_points,
		"points_current": current_points,
	}
