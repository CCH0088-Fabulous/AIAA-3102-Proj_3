PHASE_ORDER = ["part1", "part2", "part3"]
PHASE_LABELS = {
    "part1": "Part 1 Baseline",
    "part2": "Part 2 SOTA",
    "part3": "Part 3 Exploration",
}
PHASE_COLORS = {
    "part1": "#C65D3B",
    "part2": "#2A9D8F",
    "part3": "#264653",
}
PAIR_ORDER = [("part1", "part2"), ("part2", "part3"), ("part1", "part3")]
METRIC_LABELS = {
    "iou": "IoU",
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
}


def sequence_label(sequence_name):
    return sequence_name.replace("-", " ").replace("_", " ").title()