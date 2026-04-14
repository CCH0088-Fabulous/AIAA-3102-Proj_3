with open('README.md', 'r') as f:
    text = f.read()

part3_text = """
## Part 3 Progress

Part 3 introduces a state-of-the-art **Temporal Consistency Engine** combining **Dynamic Closed-Loop Masking** and **Generative Inpainting**. The current workflow includes:

- **Tracking + Refinement:** Utilizes **SAM 2** for fast temporal tracking and zero-shot instance segmentation point generation, followed immediately by **SAM 3 / SAM 3.1** zero-shot bounding-box extraction to correct edge drifts and produce pixel-perfect masks.
- **Generative Inpainting (Direction C):** Leverages **Stable Diffusion Inpainting** (via `benjamin-paine/stable-diffusion-v1-5-inpainting` from ModelScope) to generatively repair keyframe backgrounds (handling true occlusions and unseen backgrounds without warping).
- **ProPainter Temporal Propagation:** Seamlessly accepts the SD-generated highly detailed background priors and diffuses them to adjacent frames without flickering, ensuring strict temporal coherence.

Current Part 3 outputs follow this structure:
- `results/masks/part3/<sequence>/objects/`
- `results/masks/part3/<sequence>/combined/`
- `results/masks/part3/<sequence>/keyframes/` (SD-generated background scenes)
- `results/videos/part3/<sequence>_part3.mp4`

"""

if "## Part 3 Progress" not in text:
    part2_index = text.find("Current Part 2 outputs follow this structure:")
    insert_index = text.find("\n\n### 1. Run pipeline", part2_index)
    if insert_index != -1:
        text = text[:insert_index] + "\n" + part3_text + "\n" + text[insert_index:]
    else:
        text += part3_text

usage_text = """
### Example: Run Part 3 on Sequences

```bash
conda activate project3
python src/part3_exploration/pipeline_part3.py \
        --sequence bmx-trees \
        --common-config configs/common.yaml \
        --phase-config configs/part3_exploration.yaml
```

```bash
conda activate project3
python src/part3_exploration/pipeline_part3.py \
        --sequence tennis \
        --common-config configs/common.yaml \
        --phase-config configs/part3_exploration.yaml
```

"""

if "Example: Run Part 3 on Sequences" not in text:
    text += usage_text

with open('README.md', 'w') as f:
    f.write(text)
