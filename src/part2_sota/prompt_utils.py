import os
import numpy as np
from PIL import Image

def parse_prompts(prompts_args):
    prompts = []
    if not prompts_args:
        return prompts

    idx = 0
    while idx < len(prompts_args):
        frame_idx = int(prompts_args[idx]); idx += 1
        obj_id = int(prompts_args[idx]); idx += 1
        if idx < len(prompts_args) and prompts_args[idx].lower() == "box":
            idx += 1
            if idx + 4 > len(prompts_args):
                raise ValueError("Box prompt requires 4 values: x0 y0 x1 y1")
            x0 = float(prompts_args[idx]); y0 = float(prompts_args[idx + 1])
            x1 = float(prompts_args[idx + 2]); y1 = float(prompts_args[idx + 3])
            idx += 4
            prompts.append(
                {
                    "frame_idx": frame_idx,
                    "obj_id": obj_id,
                    "box": np.array([x0, y0, x1, y1], dtype=np.float32),
                }
            )
        else:
            if idx + 3 >= len(prompts_args):
                raise ValueError("Each point prompt must contain 5 values: frame_idx obj_id x y label")
            x = float(prompts_args[idx]); y = float(prompts_args[idx + 1])
            label = int(prompts_args[idx + 2])
            idx += 3
            prompts.append(
                {
                    "frame_idx": frame_idx,
                    "obj_id": obj_id,
                    "points": np.array([[x, y]], dtype=np.float32),
                    "labels": np.array([label], dtype=np.int32),
                }
            )
    return prompts


def list_image_files(image_dir):
    supported = {".jpg", ".jpeg", ".png"}
    return sorted(
        [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if os.path.splitext(fname)[1].lower() in supported
        ]
    )


def resolve_prompts(prompts_args, phase_cfg, sequence_name, input_frames_dir):
    prompts = parse_prompts(prompts_args)
    if prompts:
        return prompts

    config_prompts = phase_cfg.get("prompts", {}).get(sequence_name, [])
    if config_prompts:
        prompts = []
        for p in config_prompts:
            prompt = {
                "frame_idx": p["frame_idx"],
                "obj_id": p["obj_id"],
            }
            if "box" in p:
                prompt["box"] = np.array(p["box"], dtype=np.float32)
            elif "points" in p and "labels" in p:
                prompt["points"] = np.array(p["points"], dtype=np.float32)
                prompt["labels"] = np.array(p["labels"], dtype=np.int32)
            prompts.append(prompt)
        print(f"Using configured prompts for {sequence_name} from config file.")
        return prompts

    # Fallback to default prompts based on sequence name
    frame_files = list_image_files(input_frames_dir)
    if not frame_files:
        raise FileNotFoundError(f"No image frames found in {input_frames_dir}")
    first_image = Image.open(frame_files[0])
    width, height = first_image.size
    if "bmx-trees" in sequence_name:
        prompts = [
            {
                "frame_idx": 0,
                "obj_id": 1,
                "box": np.array([100.0, 40.0, 380.0, 220.0], dtype=np.float32),
            }
        ]
        print("Using default bmx-trees box prompt for SAM2 mask generation.")
    elif "tennis" in sequence_name:
        prompts = [
            {
                "frame_idx": 0,
                "obj_id": 1,
                "box": np.array([40.0, 40.0, 180.0, 215.0], dtype=np.float32),
                "points": np.array([[90.0, 120.0]], dtype=np.float32),
                "labels": np.array([1], dtype=np.int32),
            },
            {
                "frame_idx": 0,
                "obj_id": 2,
                "box": np.array([160.0, 70.0, 240.0, 160.0], dtype=np.float32),
                "points": np.array([[205.0, 120.0]], dtype=np.float32),
                "labels": np.array([1], dtype=np.int32),
            },
            {
                "frame_idx": 0,
                "obj_id": 3,
                "box": np.array([295.0, 145.0, 325.0, 175.0], dtype=np.float32),
                "points": np.array([[310.0, 160.0]], dtype=np.float32),
                "labels": np.array([1], dtype=np.int32),
            },
            {
                "frame_idx": 0,
                "obj_id": 4,
                "box": np.array([20.0, 190.0, 150.0, 240.0], dtype=np.float32),
                "points": np.array([[70.0, 220.0]], dtype=np.float32),
                "labels": np.array([1], dtype=np.int32),
            },
            {
                "frame_idx": 0,
                "obj_id": 5,
                "box": np.array([220.0, 205.0, 325.0, 240.0], dtype=np.float32),
                "points": np.array([[255.0, 225.0]], dtype=np.float32),
                "labels": np.array([1], dtype=np.int32),
            }
        ]
        print("Using default tennis box+point prompts for SAM2 mask generation (person, racket, ball, shadows).")
    else:
        prompts = [
            {
                "frame_idx": 0,
                "obj_id": 1,
                "points": np.array([[width / 2.0, height / 2.0]], dtype=np.float32),
                "labels": np.array([1], dtype=np.int32),
            }
        ]
        print(f"Using default center point prompt for {sequence_name} (no specific config found).")
    return prompts
