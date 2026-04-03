import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from common.optical_flow import estimate_mask_motion


class DynamicObjectJudge:
    """Filter candidate masks by sparse Lucas-Kanade motion statistics."""

    def __init__(
        self,
        motion_threshold,
        min_tracked_points,
        aggregation="median",
        keep_if_undetermined=True,
        feature_params=None,
        lk_params=None,
    ):
        self.motion_threshold = float(motion_threshold)
        self.min_tracked_points = int(min_tracked_points)
        self.aggregation = aggregation
        self.keep_if_undetermined = keep_if_undetermined
        self.feature_params = feature_params or {}
        self.lk_params = lk_params or {}

    def _select_motion_score(self, motion_summary):
        if self.aggregation == "mean":
            return motion_summary["mean_motion"]
        return motion_summary["median_motion"]

    def evaluate_instance(self, previous_frame, current_frame, mask):
        if previous_frame is None:
            return {
                "selected": True,
                "is_dynamic": True,
                "reason": "no_previous_frame",
                "score": None,
                "valid": False,
                "num_features": 0,
                "num_tracked": 0,
            }

        motion_summary = estimate_mask_motion(
            previous_frame,
            current_frame,
            mask,
            feature_params=self.feature_params,
            lk_params=self.lk_params,
            min_tracked_points=self.min_tracked_points,
        )
        score = self._select_motion_score(motion_summary)

        if motion_summary["valid"]:
            is_dynamic = score >= self.motion_threshold
            reason = "motion_above_threshold" if is_dynamic else "motion_below_threshold"
        else:
            is_dynamic = bool(self.keep_if_undetermined)
            reason = "insufficient_tracked_points"

        motion_summary.update(
            {
                "selected": is_dynamic,
                "is_dynamic": is_dynamic,
                "reason": reason,
                "score": float(score),
            }
        )
        return motion_summary

    def filter_dynamic_instances(self, previous_frame, current_frame, masks):
        if not masks:
            return [], []

        filtered_masks = []
        motion_summaries = []

        for mask in masks:
            motion_summary = self.evaluate_instance(previous_frame, current_frame, mask)
            motion_summaries.append(motion_summary)
            if motion_summary["selected"]:
                filtered_masks.append(mask)

        return filtered_masks, motion_summaries