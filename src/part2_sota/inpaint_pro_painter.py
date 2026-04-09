import os
import sys
import numpy as np
from pathlib import Path

# Fix OpenCV import path for conda-installed cv2
def _ensure_cv2_path():
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cv2_dir = Path(sys.prefix) / "lib" / py_ver / "site-packages" / "cv2" / f"python-{sys.version_info.major}.{sys.version_info.minor}"
    if cv2_dir.exists() and str(cv2_dir) not in sys.path:
        sys.path.insert(0, str(cv2_dir))

_ensure_cv2_path()
import cv2
import torch
import imageio
from PIL import Image

# Add ProPainter to path
PROP_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "ProPainter"
if str(PROP_DIR) not in sys.path:
    sys.path.append(str(PROP_DIR))

from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device


class ProPainterInpainter:
    """ProPainter-based video inpainting."""

    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.device = get_device()
        self.raft_model = None
        self.flow_comp_model = None
        self.inpainter = None
        self._load_models()

    def _compute_process_size(self, height, width, video_length):
        """Choose a memory-friendly processing size while preserving aspect ratio."""
        max_side = max(height, width)
        # Long videos consume a lot of VRAM in RAFT flow estimation.
        if video_length >= 120:
            target_max_side = 320
        elif video_length >= 80:
            target_max_side = 384
        else:
            target_max_side = 512

        if max_side <= target_max_side:
            return height, width

        scale = target_max_side / float(max_side)
        resized_h = max(32, int(round(height * scale / 8.0) * 8))
        resized_w = max(32, int(round(width * scale / 8.0) * 8))
        return resized_h, resized_w

    def _load_models(self):
        """Load ProPainter models lazily to avoid import-time cv2/torch conflicts."""
        from model.modules.flow_comp_raft import RAFT_bi
        from model.recurrent_flow_completion import RecurrentFlowCompleteNet
        from model.propainter import InpaintGenerator

        # Load RAFT
        raft_ckpt = load_file_from_url(
            'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth',
            str(self.weights_dir / 'weights')
        )
        self.raft_model = RAFT_bi(raft_ckpt, self.device)

        # Load flow completion
        flow_comp_ckpt = load_file_from_url(
            'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth',
            str(self.weights_dir / 'weights')
        )
        self.flow_comp_model = RecurrentFlowCompleteNet(flow_comp_ckpt)
        self.flow_comp_model.eval().to(self.device)

        # Load ProPainter
        propainter_ckpt = load_file_from_url(
            'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth',
            str(self.weights_dir / 'weights')
        )
        self.inpainter = InpaintGenerator(model_path=propainter_ckpt).to(self.device)
        self.inpainter.eval()

    def _resolve_mask_path(self, masks_dir, frame_filename):
        frame_stem = Path(frame_filename).stem
        candidates = [
            os.path.join(masks_dir, frame_filename),
            os.path.join(masks_dir, f"{frame_stem}.png"),
            os.path.join(masks_dir, f"{frame_stem}.jpg"),
            os.path.join(masks_dir, f"{frame_stem}.jpeg"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def inpaint(self, frames_dir, masks_dir, output_video_path=None, fps=24):
        """
        Inpaint video frames using masks.

        Args:
            frames_dir (str): Directory with input frames.
            masks_dir (str): Directory with mask frames (grayscale, same names as frames).
            output_video_path (str): Path to save output video. If None, returns frames.
            fps (int): FPS for output video.

        Returns:
            list: List of inpainted frames as numpy arrays, or saves video if output_video_path provided.
        """
        # Load frames
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        frames = []
        for f in frame_files:
            img = Image.open(os.path.join(frames_dir, f)).convert('RGB')
            frames.append(img)  # Keep as PIL Images for to_tensors()

        # Load masks
        masks = []
        for f in frame_files:
            mask_path = self._resolve_mask_path(masks_dir, f)
            if mask_path:
                mask = Image.open(mask_path).convert('L')
                masks.append(np.array(mask) / 255.0)  # Normalize to 0-1
            else:
                # Create zero mask with same size as first frame
                first_frame = frames[0]
                masks.append(np.zeros((first_frame.height, first_frame.width)))

        # Get original size
        h, w = frames[0].height, frames[0].width
        process_h, process_w = self._compute_process_size(h, w, len(frames))
        process_size = (process_h, process_w)

        # Resize frames/masks for memory-efficient inpainting, then upscale outputs back.
        if (process_h, process_w) != (h, w):
            resized_frames = [img.resize((process_w, process_h), Image.BILINEAR) for img in frames]
            resized_masks = [
                cv2.resize(mask.astype(np.float32), (process_w, process_h), interpolation=cv2.INTER_NEAREST)
                for mask in masks
            ]
        else:
            resized_frames = frames
            resized_masks = masks

        # Dilate masks
        kernel = np.ones((4, 4), np.uint8)
        masks_dilated = [cv2.dilate((m > 0).astype(np.uint8), kernel) for m in resized_masks]
        flow_masks = [cv2.dilate((m > 0).astype(np.uint8), kernel) for m in resized_masks]

        # Convert masks to PIL Images for to_tensors()
        masks_dilated_pil = [Image.fromarray((m * 255).astype(np.uint8), mode='L') for m in masks_dilated]
        flow_masks_pil = [Image.fromarray((m * 255).astype(np.uint8), mode='L') for m in flow_masks]

        # Convert to tensors
        frames_tensor = to_tensors()(resized_frames).unsqueeze(0) * 2 - 1  # (1, 3, L, H, W)
        frames_tensor = frames_tensor.to(self.device)
        flow_masks_tensor = to_tensors()(flow_masks_pil).unsqueeze(0)  # (1, 1, L, H, W)
        flow_masks_tensor = flow_masks_tensor.to(self.device)
        masks_dilated_tensor = to_tensors()(masks_dilated_pil).unsqueeze(0)  # (1, 1, L, H, W)
        masks_dilated_tensor = masks_dilated_tensor.to(self.device)

        video_length = frames_tensor.size(1)

        # Compute flows
        with torch.no_grad():
            gt_flows_bi = self.raft_model(frames_tensor, iters=20)

            # Complete flows
            pred_flows_bi, _ = self.flow_comp_model.forward_bidirect_flow(gt_flows_bi, flow_masks_tensor)
            pred_flows_bi = self.flow_comp_model.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks_tensor)

            # Image propagation
            masked_frames = frames_tensor * (1 - masks_dilated_tensor)
            prop_imgs, updated_masks = self.inpainter.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated_tensor, 'nearest'
            )
            updated_frames = frames_tensor * (1 - masks_dilated_tensor) + \
                           prop_imgs * masks_dilated_tensor

            # Inpainting
            comp_frames = [None] * video_length
            neighbor_stride = 10
            for f in range(0, video_length, neighbor_stride):
                neighbor_ids = list(range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1)))
                ref_ids = list(range(0, video_length, 10))
                selected_imgs = updated_frames[:, neighbor_ids + ref_ids]
                selected_masks = masks_dilated_tensor[:, neighbor_ids + ref_ids]
                selected_updated_masks = updated_masks[:, neighbor_ids + ref_ids]
                selected_flows = (pred_flows_bi[0][:, neighbor_ids[:-1]], pred_flows_bi[1][:, neighbor_ids[:-1]])

                l_t = len(neighbor_ids)
                pred_img = self.inpainter(selected_imgs, selected_flows, selected_masks, selected_updated_masks, l_t)
                pred_img = pred_img.view(-1, 3, *process_size)
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255

                binary_masks = masks_dilated_tensor[0, neighbor_ids, 0].cpu().numpy().astype(np.uint8)

                for i, idx in enumerate(neighbor_ids):
                    mask_expanded = binary_masks[i, :, :, np.newaxis]  # (H, W, 1)
                    img = pred_img[i] * mask_expanded + \
                          np.array(resized_frames[idx]) * (1 - mask_expanded)
                    comp_frames[idx] = img.astype(np.uint8).squeeze()

            # Ensure frames are ordered and filled
            if any(frame is None for frame in comp_frames):
                raise RuntimeError("Some inpainted frames were not generated; check neighbor window logic.")

        # Resize back
        comp_frames_resized = [cv2.resize(f, (w, h)) for f in comp_frames]

        if output_video_path:
            imageio.mimwrite(output_video_path, comp_frames_resized, fps=fps)

        return comp_frames_resized