import math

import numpy as np
from scipy.ndimage import gaussian_filter


def _ensure_same_shape(array_a, array_b, name_a="array_a", name_b="array_b"):
	if array_a.shape != array_b.shape:
		raise ValueError(
			f"Expected {name_a} and {name_b} to have the same shape, "
			f"but got {array_a.shape} and {array_b.shape}."
		)


def normalize_binary_mask(mask):
    if mask is None:
        raise ValueError("Expected a valid mask array, but received None.")
    if mask.ndim == 3:
        mask = np.mean(mask, axis=2)
    return (mask > 0).astype(np.uint8)


def compute_iou(predicted_mask, reference_mask):
	predicted = normalize_binary_mask(predicted_mask).astype(bool)
	reference = normalize_binary_mask(reference_mask).astype(bool)
	_ensure_same_shape(predicted, reference, "predicted_mask", "reference_mask")

	intersection = int(np.logical_and(predicted, reference).sum())
	union = int(np.logical_or(predicted, reference).sum())
	iou = 1.0 if union == 0 else intersection / union
	return {
		"iou": float(iou),
		"intersection": intersection,
		"union": union,
	}


def build_background_valid_mask(predicted_mask, reference_mask=None):
	if predicted_mask is None and reference_mask is None:
		raise ValueError("Expected at least one mask to build background valid mask.")

	if predicted_mask is None:
		predicted = None
	else:
		predicted = normalize_binary_mask(predicted_mask).astype(bool)

	if reference_mask is None:
		reference = None
	else:
		reference = normalize_binary_mask(reference_mask).astype(bool)

	if predicted is None:
		combined_foreground = reference
	elif reference is None:
		combined_foreground = predicted
	else:
		_ensure_same_shape(predicted, reference, "predicted_mask", "reference_mask")
		combined_foreground = np.logical_or(predicted, reference)

	return (~combined_foreground).astype(np.uint8)


def compute_psnr(reference_image, compared_image, valid_mask=None, data_range=255.0):
	_ensure_same_shape(reference_image, compared_image, "reference_image", "compared_image")

	reference = reference_image.astype(np.float64)
	compared = compared_image.astype(np.float64)
	squared_error = (reference - compared) ** 2

	if valid_mask is not None:
		valid = normalize_binary_mask(valid_mask).astype(bool)
		_ensure_same_shape(valid, reference[..., 0] if reference.ndim == 3 else reference, "valid_mask", "reference")
		if not np.any(valid):
			return float("nan")
		squared_error = squared_error[valid]

	mse = float(np.mean(squared_error)) if squared_error.size else float("nan")
	if math.isnan(mse):
		return float("nan")
	if mse == 0.0:
		return float("inf")
	return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def _prepare_for_ssim(image):
	prepared = image.astype(np.float64)
	if prepared.ndim == 2:
		prepared = prepared[..., None]
	return prepared


def compute_ssim(reference_image, compared_image, valid_mask=None, data_range=255.0):
	_ensure_same_shape(reference_image, compared_image, "reference_image", "compared_image")

	reference = _prepare_for_ssim(reference_image)
	compared = _prepare_for_ssim(compared_image)

	kernel_size = (11, 11)
	sigma = 1.5
	c1 = (0.01 * data_range) ** 2
	c2 = (0.03 * data_range) ** 2

	ssim_maps = []
	for channel_index in range(reference.shape[2]):
		channel_reference = reference[..., channel_index]
		channel_compared = compared[..., channel_index]

		truncate = (kernel_size[0] - 1) / (2.0 * sigma)
		mu_reference = gaussian_filter(channel_reference, sigma=sigma, truncate=truncate)
		mu_compared = gaussian_filter(channel_compared, sigma=sigma, truncate=truncate)

		mu_reference_sq = mu_reference ** 2
		mu_compared_sq = mu_compared ** 2
		mu_reference_compared = mu_reference * mu_compared

		sigma_reference_sq = gaussian_filter(channel_reference ** 2, sigma=sigma, truncate=truncate) - mu_reference_sq
		sigma_compared_sq = gaussian_filter(channel_compared ** 2, sigma=sigma, truncate=truncate) - mu_compared_sq
		sigma_reference_compared = (
			gaussian_filter(channel_reference * channel_compared, sigma=sigma, truncate=truncate)
			- mu_reference_compared
		)

		numerator = (2 * mu_reference_compared + c1) * (2 * sigma_reference_compared + c2)
		denominator = (mu_reference_sq + mu_compared_sq + c1) * (sigma_reference_sq + sigma_compared_sq + c2)
		ssim_maps.append(numerator / (denominator + 1e-12))

	ssim_map = np.mean(np.stack(ssim_maps, axis=0), axis=0)

	if valid_mask is not None:
		valid = normalize_binary_mask(valid_mask).astype(bool)
		_ensure_same_shape(valid, ssim_map, "valid_mask", "ssim_map")
		if not np.any(valid):
			return float("nan")
		return float(np.mean(ssim_map[valid]))

	return float(np.mean(ssim_map))
