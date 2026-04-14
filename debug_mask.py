import cv2
import numpy as np
from src.common.config import load_yaml_config

cfg = load_yaml_config("configs/part1_baseline.yaml")
print(cfg.get("pipeline", {}).get("postprocess"))
