import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image


class SAM3UpgradeRefiner:
    def __init__(
        self,
        weights_dir,
        checkpoint_dir=None,
        checkpoint_name=None,
        repo_id=None,
        auto_download=True,
        confidence_threshold=0.35,
        box_expand_ratio=0.08,
        min_mask_area=32,
        min_selection_iou=0.05,
        min_area_ratio=0.95,
        max_area_ratio=1.05,
        min_candidate_precision=0.95,
        min_coarse_recall=0.95,
        fallback_to_coarse=True,
    ):
        self.weights_dir = Path(weights_dir)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.weights_dir / "checkpoints"
        self.checkpoint_name = checkpoint_name
        self.repo_id = repo_id
        self.auto_download = auto_download
        self.confidence_threshold = confidence_threshold
        self.box_expand_ratio = box_expand_ratio
        self.min_mask_area = int(min_mask_area)
        self.min_selection_iou = float(min_selection_iou)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.min_candidate_precision = float(min_candidate_precision)
        self.min_coarse_recall = float(min_coarse_recall)
        self.fallback_to_coarse = fallback_to_coarse
        self.device = "cuda"
        self.model = None
        self.processor = None
        self.checkpoint_path = None
        self._load_model()

    def _resolve_checkpoint(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        candidates = []
        if self.checkpoint_name:
            candidates.append((self.repo_id or self._infer_repo_id(self.checkpoint_name), self.checkpoint_name))
        else:
            candidates.extend(
                [
                    ("facebook/sam3.1", "sam3.1_multiplex.pt"),
                    ("facebook/sam3", "sam3.pt"),
                ]
            )

        for _, checkpoint_name in candidates:
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            if checkpoint_path.exists():
                return checkpoint_path

        if not self.auto_download:
            raise FileNotFoundError(
                f"No SAM3 checkpoint found in {self.checkpoint_dir}. "
                f"Expected one of: {', '.join(name for _, name in candidates)}"
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to auto-download SAM3 checkpoints.") from exc

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        errors = []
        for repo_id, checkpoint_name in candidates:
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=checkpoint_name,
                    local_dir=self.checkpoint_dir,
                    token=token,
                )
                checkpoint_path = self.checkpoint_dir / checkpoint_name
                if checkpoint_path.exists():
                    return checkpoint_path
            except Exception as exc:
                errors.append(f"{repo_id}/{checkpoint_name}: {exc}")

        raise RuntimeError(
            "Unable to download a SAM3 checkpoint into "
            f"{self.checkpoint_dir}. Ensure this machine can reach Hugging Face and that your token "
            f"has access to the gated repo. Attempts: {' | '.join(errors)}"
        )

    def _infer_repo_id(self, checkpoint_name):
        if checkpoint_name.startswith("sam3.1"):
            return "facebook/sam3.1"
        return "facebook/sam3"

    def _load_model(self):
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = self._resolve_checkpoint()
        self.model = build_sam3_image_model(
            checkpoint_path=str(self.checkpoint_path),
            load_from_HF=False,
            device=self.device,
        )
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )

    def _mask_to_bbox_xywh(self, mask):
        binary_mask = self._normalize_mask(mask)
        if binary_mask is None:
            return None

        ys, xs = np.where(binary_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None

        x0 = int(xs.min())
        x1 = int(xs.max())
        y0 = int(ys.min())
        y1 = int(ys.max())

        width = x1 - x0 + 1
        height = y1 - y0 + 1
        expand_x = int(round(width * self.box_expand_ratio))
        expand_y = int(round(height * self.box_expand_ratio))

        x0 = max(0, x0 - expand_x)
        y0 = max(0, y0 - expand_y)
        x1 = min(binary_mask.shape[1] - 1, x1 + expand_x)
        y1 = min(binary_mask.shape[0] - 1, y1 + expand_y)

        return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]

    def _normalize_mask(self, mask):
        if mask is None:
            return None
        binary_mask = ((np.asarray(mask) > 0).astype(np.uint8)) * 255
        if binary_mask.ndim != 2:
            return None
        if int(binary_mask.sum() / 255) < self.min_mask_area:
            return None
        return binary_mask

    def _tensor_mask_to_numpy(self, mask_tensor):
        mask = mask_tensor.detach().cpu().numpy()
        while mask.ndim > 2:
            mask = mask.squeeze(0)
        return (mask > 0).astype(np.uint8) * 255

    def _compute_iou(self, first_mask, second_mask):
        first_binary = first_mask > 0
        second_binary = second_mask > 0
        union = np.logical_or(first_binary, second_binary).sum()
        if union == 0:
            return 0.0
        intersection = np.logical_and(first_binary, second_binary).sum()
        return float(intersection / union)

    def _compute_overlap_stats(self, candidate_mask, coarse_mask):
        candidate_binary = candidate_mask > 0
        coarse_binary = coarse_mask > 0
        intersection = int(np.logical_and(candidate_binary, coarse_binary).sum())
        candidate_area = int(candidate_binary.sum())
        coarse_area = int(coarse_binary.sum())
        union = int(np.logical_or(candidate_binary, coarse_binary).sum())

        coarse_iou = float(intersection / union) if union else 0.0
        area_ratio = float(candidate_area / coarse_area) if coarse_area else 0.0
        candidate_precision = float(intersection / candidate_area) if candidate_area else 0.0
        coarse_recall = float(intersection / coarse_area) if coarse_area else 0.0
        return {
            "intersection": intersection,
            "candidate_area": candidate_area,
            "coarse_area": coarse_area,
            "coarse_iou": coarse_iou,
            "area_ratio": area_ratio,
            "candidate_precision": candidate_precision,
            "coarse_recall": coarse_recall,
        }

    def _passes_consistency_gate(self, overlap_stats):
        return (
            overlap_stats["coarse_iou"] >= self.min_selection_iou
            and self.min_area_ratio <= overlap_stats["area_ratio"] <= self.max_area_ratio
            and overlap_stats["candidate_precision"] >= self.min_candidate_precision
            and overlap_stats["coarse_recall"] >= self.min_coarse_recall
        )

    def _normalize_bbox_xywh(self, box_xywh, image_width, image_height):
        normalized_box = list(box_xywh)
        normalized_box[0] /= float(image_width)
        normalized_box[1] /= float(image_height)
        normalized_box[2] /= float(image_width)
        normalized_box[3] /= float(image_height)
        return normalized_box

    def _run_frame_refinement(self, frame_path, coarse_mask):
        import torch
        from sam3.model.box_ops import box_xywh_to_cxcywh

        binary_mask = self._normalize_mask(coarse_mask)
        if binary_mask is None:
            return [], np.zeros_like(np.asarray(coarse_mask, dtype=np.uint8))

        image = Image.open(frame_path).convert("RGB")
        width, height = image.size
        box_xywh = self._mask_to_bbox_xywh(binary_mask)
        if box_xywh is None:
            return [], binary_mask if self.fallback_to_coarse else np.zeros_like(binary_mask)

        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else nullcontext()
        )
        with autocast_context:
            state = self.processor.set_image(image)
            box_input_xywh = torch.tensor(box_xywh, dtype=torch.float32).view(-1, 4)
            box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
            norm_box_cxcywh = self._normalize_bbox_xywh(box_input_cxcywh.flatten().tolist(), width, height)
            state = self.processor.add_geometric_prompt(state=state, box=norm_box_cxcywh, label=True)

        candidate_masks = []
        candidate_scores = state.get("scores", [])
        for index, mask_tensor in enumerate(state.get("masks", [])):
            candidate_mask = self._tensor_mask_to_numpy(mask_tensor)
            overlap_stats = self._compute_overlap_stats(candidate_mask, binary_mask)
            score_value = float(candidate_scores[index].detach().cpu().item()) if len(candidate_scores) > index else 0.0
            candidate_masks.append(
                {
                    "mask": candidate_mask,
                    "score": score_value,
                    **overlap_stats,
                }
            )

        if not candidate_masks:
            fallback_mask = binary_mask if self.fallback_to_coarse else np.zeros_like(binary_mask)
            return [], fallback_mask

        best_candidate = max(
            candidate_masks,
            key=lambda item: (
                self._passes_consistency_gate(item),
                item["coarse_iou"],
                item["candidate_precision"],
                item["coarse_recall"],
                -abs(1.0 - item["area_ratio"]),
                item["score"],
            ),
        )
        if not self._passes_consistency_gate(best_candidate) and self.fallback_to_coarse:
            refined_mask = binary_mask
        else:
            refined_mask = best_candidate["mask"]

        return candidate_masks, refined_mask

    def refine(self, frame_paths, coarse_masks):
        candidate_masks_per_frame = []
        refined_masks = []

        for frame_path, coarse_mask in zip(frame_paths, coarse_masks):
            candidates, refined_mask = self._run_frame_refinement(frame_path, coarse_mask)
            candidate_masks_per_frame.append(candidates)
            refined_masks.append(refined_mask)

        return candidate_masks_per_frame, refined_masks