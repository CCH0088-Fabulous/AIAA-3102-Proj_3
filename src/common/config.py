import os

import yaml


def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_image_extensions(common_cfg):
    extensions = common_cfg.get("naming", {}).get("image_extensions", [])
    return tuple(extensions) if extensions else (".jpg", ".jpeg", ".png")


def collect_image_paths(dataset_dir, common_cfg):
    image_paths = []
    for extension in get_image_extensions(common_cfg):
        suffix = extension if extension.startswith(".") else f".{extension}"
        pattern = f"*{suffix}"
        image_paths.extend(
            os.path.join(dataset_dir, filename)
            for filename in os.listdir(dataset_dir)
            if filename.lower().endswith(suffix.lower())
        )
    return sorted(set(image_paths))


def resolve_image_dir(candidate_dir, common_cfg):
    if not candidate_dir or not os.path.isdir(candidate_dir):
        return None

    if collect_image_paths(candidate_dir, common_cfg):
        return candidate_dir

    nested_dir = os.path.join(candidate_dir, os.path.basename(os.path.normpath(candidate_dir)))
    if os.path.isdir(nested_dir) and collect_image_paths(nested_dir, common_cfg):
        return nested_dir

    return None


def normalize_sequence_key(sequence_key, common_cfg):
    aliases = common_cfg.get("datasets", {}).get("aliases", {})
    return aliases.get(sequence_key, sequence_key)


def resolve_sequence_spec(sequence_arg, common_cfg):
    datasets_cfg = common_cfg.get("datasets", {})
    canonical_name = normalize_sequence_key(sequence_arg, common_cfg)
    sequences_cfg = datasets_cfg.get("sequences", {})

    if canonical_name in sequences_cfg:
        sequence_spec = dict(sequences_cfg[canonical_name])
        resolved_frames_dir = resolve_image_dir(sequence_spec.get("frames_dir"), common_cfg)
        if not resolved_frames_dir:
            raise FileNotFoundError(
                f"Configured dataset path does not exist or has no frames: {sequence_spec.get('frames_dir')}"
            )
        sequence_spec["frames_dir"] = resolved_frames_dir
        sequence_spec["canonical_name"] = canonical_name
        sequence_spec.setdefault("output_name", canonical_name)
        return sequence_spec

    candidate_paths = [
        sequence_arg,
        os.path.join("data", "raw", sequence_arg, sequence_arg),
        os.path.join("data", "raw", sequence_arg),
    ]

    davis_jpeg_root = datasets_cfg.get("davis", {}).get("jpeg_root")
    if davis_jpeg_root:
        candidate_paths.append(os.path.join(davis_jpeg_root, sequence_arg))

    for candidate in candidate_paths:
        resolved_frames_dir = resolve_image_dir(candidate, common_cfg)
        if resolved_frames_dir:
            output_name = os.path.basename(os.path.normpath(resolved_frames_dir))
            return {
                "canonical_name": output_name,
                "output_name": output_name,
                "frames_dir": resolved_frames_dir,
                "reference_masks_dir": None,
            }

    raise FileNotFoundError(
        f"Unable to resolve dataset sequence '{sequence_arg}'. Pass a valid dataset key or a folder containing frames."
    )


def get_phase_output_dir(phase_cfg, output_key):
    return phase_cfg.get("phase", {}).get("outputs", {}).get(output_key)


def ensure_phase_output_dirs(phase_cfg):
    outputs = phase_cfg.get("phase", {}).get("outputs", {})
    for output_dir in outputs.values():
        os.makedirs(output_dir, exist_ok=True)


def build_frame_filename(common_cfg, frame_index, artifact="mask"):
    naming_cfg = common_cfg.get("naming", {})
    if artifact == "frame":
        pattern = naming_cfg.get("frame_file_pattern", "frame_{frame_index:04d}.png")
    else:
        pattern = naming_cfg.get("mask_file_pattern", "frame_{frame_index:04d}.png")
    return pattern.format(frame_index=frame_index)


def build_video_filename(common_cfg, sequence_name, phase_slug):
    pattern = common_cfg.get("naming", {}).get(
        "video_file_pattern",
        "{sequence_name}_{phase_slug}.mp4",
    )
    return pattern.format(sequence_name=sequence_name, phase_slug=phase_slug)