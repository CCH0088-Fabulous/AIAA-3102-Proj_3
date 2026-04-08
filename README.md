# Video Object Removal and Inpainting

This repository implements a staged video object removal project. The current completed work is Part 1, a classical baseline pipeline built around YOLOv8-Seg, sparse optical flow, temporal background propagation, and traditional image inpainting.

Part 1 is no longer just a mask extraction demo. It now runs a full baseline pipeline:

1. Detect candidate dynamic classes with YOLOv8-Seg.
2. Use Lucas-Kanade sparse optical flow to decide whether each detected instance is actually moving.
3. Postprocess masks with dilation, hole filling, connected-component cleanup, and short temporal smoothing.
4. Restore the video by borrowing clean background pixels from nearby frames first.
5. Fill any remaining holes with `cv2.inpaint` using Telea or Navier-Stokes.
6. Export masks, restored video, visualization artifacts, and evaluation metrics.

## Current Status

- Part 1 baseline: implemented and validated.
- Part 2 SOTA pipeline: interface scaffold only.
- Part 3 exploration pipeline: interface scaffold only.

## Environment

The commands below assume the Conda environment is named `project3`.

```bash
conda activate project3
pip install -r requirements.txt
```

Expected Part 1 model weight path:

```text
models/yolo_v8_seg/yolov8x-seg.pt
```

If the weight file is missing, place the YOLOv8 segmentation checkpoint at that location or update [configs/part1_baseline.yaml](configs/part1_baseline.yaml).

## Dataset Keys

Part 1 currently supports dataset keys defined in [configs/common.yaml](configs/common.yaml):

- `bmx-trees`
- `tennis`
- `davis` sequences resolved from DAVIS roots

For compatibility, `bmx_trees` is also accepted and mapped to `bmx-trees` internally.

## Part 1 Method

### 1. Candidate Mask Extraction

The baseline starts with YOLOv8-Seg in [src/part1_baseline/mask_extraction_yolo.py](src/part1_baseline/mask_extraction_yolo.py). The model extracts instance masks for configured dynamic classes. The default target classes in [configs/part1_baseline.yaml](configs/part1_baseline.yaml) are:

- `0`: person
- `1`: bicycle

### 2. Dynamic Object Judgment

Detected masks are not removed blindly. Each candidate instance is passed to [src/part1_baseline/dynamic_judgment.py](src/part1_baseline/dynamic_judgment.py), which uses sparse Lucas-Kanade optical flow from [src/common/optical_flow.py](src/common/optical_flow.py) to measure motion magnitude inside the mask.

Current default behavior:

- motion aggregation: `median`
- motion threshold: `1.5`
- minimum tracked points: `8`
- if motion is undetermined because too few points are tracked, keep the instance instead of dropping it

This prevents the pipeline from deleting every detected person or bicycle when the object is actually static.

### 3. Mask Postprocessing

After dynamic filtering, the final removal mask is cleaned in [src/common/mask_utils.py](src/common/mask_utils.py). The current postprocessing chain is:

- dilation
- hole filling
- removal of small connected components
- trailing-window temporal smoothing

This improves coverage around motion blur and stabilizes masks across adjacent frames.

### 4. Traditional Restoration

The restoration stage is implemented in [src/part1_baseline/inpaint_traditional.py](src/part1_baseline/inpaint_traditional.py). It follows the Part 1 project requirement closely:

1. For each masked frame, search neighboring frames inside a configurable temporal window.
2. Copy pixels from the same spatial location only when the candidate frame is clean at that location.
3. If masked holes remain after temporal borrowing, use `cv2.inpaint` as a fallback.

Supported fallback modes:

- `telea`
- `navier-stokes`
- `ns`

### 5. Explainability and Report Assets

Part 1 also exports visualization artifacts from [src/common/visualization.py](src/common/visualization.py) to support debugging and report writing:

- per-instance motion score overlays
- final mask overlays
- original / mask overlay / restored comparison triptychs

## Part 1 Configuration

The main Part 1 config lives in [configs/part1_baseline.yaml](configs/part1_baseline.yaml). The most important blocks are:

- `models.segmentation.weights`: YOLOv8-Seg checkpoint path
- `pipeline.input.sequence_key`: default sequence
- `pipeline.dynamic_filter.*`: motion thresholding and LK parameters
- `pipeline.postprocess.*`: mask cleanup settings
- `pipeline.inpainting.*`: temporal propagation and fallback inpainting settings
- `pipeline.visualization.*`: export controls for debug/report images

Shared paths, naming rules, dataset aliases, and output roots are defined in [configs/common.yaml](configs/common.yaml).

## Part 1 Outputs

After running the baseline, the main outputs are:

- masks: `results/masks/part1/{sequence_name}/`
- restored video: `results/videos/part1/{sequence_name}_part1.mp4`
- visualizations: `results/visualizations/part1/{sequence_name}/`
- metrics: `results/metrics/iou_results.csv` and `results/metrics/psnr_ssim.csv`

For example, `bmx-trees` generates:

- `results/masks/part1/bmx-trees/`
- `results/videos/part1/bmx-trees_part1.mp4`
- `results/visualizations/part1/bmx-trees/`

## Part 1 Run Commands

### 1. Standard Part 1 Run

Run the full Part 1 pipeline on the default sequence from the config:

```bash
conda activate project3
python src/part1_baseline/pipeline_part1.py \
	--common-config configs/common.yaml \
	--phase-config configs/part1_baseline.yaml
```

### 2. Run Part 1 on a Specific Sequence Key

```bash
conda activate project3
python src/part1_baseline/pipeline_part1.py \
	--sequence bmx-trees \
	--common-config configs/common.yaml \
	--phase-config configs/part1_baseline.yaml
```

```bash
conda activate project3
python src/part1_baseline/pipeline_part1.py \
	--sequence tennis \
	--common-config configs/common.yaml \
	--phase-config configs/part1_baseline.yaml
```

### 3. Run Part 1 with a Direct Frame Folder

```bash
conda activate project3
python src/part1_baseline/pipeline_part1.py \
	--sequence /absolute/path/to/frames \
	--common-config configs/common.yaml \
	--phase-config configs/part1_baseline.yaml
```

### 4. Backward-Compatible Alias

The old `--folder` argument is still supported:

```bash
conda activate project3
python src/part1_baseline/pipeline_part1.py \
	--folder bmx_trees \
	--common-config configs/common.yaml \
	--phase-config configs/part1_baseline.yaml
```

### 5. Quick Smoke Test

Limit processing to a few frames for debugging:

```bash
conda activate project3
python src/part1_baseline/pipeline_part1.py \
	--sequence bmx-trees \
	--max-frames 3 \
	--common-config configs/common.yaml \
	--phase-config configs/part1_baseline.yaml
```

### 6. Run via Shell Wrapper

The repository also provides a wrapper script:

```bash
conda activate project3
bash scripts/run_part1.sh --sequence bmx-trees --max-frames 3
```

## Part 1 Evaluation Commands

### 1. Default Evaluation

This evaluates:

- IoU against dataset reference masks when available
- PSNR and SSIM in `background_preservation` mode when no clean reference frames are provided

```bash
conda activate project3
python scripts/evaluate_metrics.py \
	--phase-config configs/part1_baseline.yaml \
	--sequence bmx-trees
```

### 2. Quick Evaluation on a Few Frames

```bash
conda activate project3
python scripts/evaluate_metrics.py \
	--phase-config configs/part1_baseline.yaml \
	--sequence bmx-trees \
	--max-frames 3
```

### 3. Full-Reference PSNR and SSIM

If you have clean target frames for the same sequence, pass them explicitly:

```bash
conda activate project3
python scripts/evaluate_metrics.py \
	--phase-config configs/part1_baseline.yaml \
	--sequence bmx-trees \
	--reference-frames-dir /absolute/path/to/clean_reference_frames
```

In this mode, the script switches to `full_reference` evaluation for PSNR and SSIM.

### 4. Custom Evaluation Inputs

You can override any evaluation source manually:

```bash
conda activate project3
python scripts/evaluate_metrics.py \
	--phase-config configs/part1_baseline.yaml \
	--sequence bmx-trees \
	--mask-dir results/masks/part1/bmx-trees \
	--video-path results/videos/part1/bmx-trees_part1.mp4 \
	--reference-mask-dir data/raw/bmx-trees/bmx-trees_mask
```

## Visualization Outputs

Part 1 explainability outputs are exported to:

```text
results/visualizations/part1/{sequence_name}/
```

Each sequence contains:

- `motion_scores/`: instance-level motion score overlays with dynamic/static decisions
- `mask_overlays/`: final removal mask overlaid on source frames
- `comparisons/`: original / mask overlay / restored comparison frames

Example validated output directories:

- `results/visualizations/part1/bmx-trees/motion_scores/`
- `results/visualizations/part1/bmx-trees/mask_overlays/`
- `results/visualizations/part1/bmx-trees/comparisons/`

## Example Part 1 Workflow

Run the pipeline, generate visualizations, and evaluate metrics for `bmx-trees`:

```bash
conda activate project3
python src/part1_baseline/pipeline_part1.py \
	--sequence bmx-trees \
	--common-config configs/common.yaml \
	--phase-config configs/part1_baseline.yaml

python scripts/evaluate_metrics.py \
	--phase-config configs/part1_baseline.yaml \
	--sequence bmx-trees
```

## Current Validated Part 1 Artifacts

The current repository state has already produced and validated these outputs:

- restored video: [results/videos/part1/bmx-trees_part1.mp4](results/videos/part1/bmx-trees_part1.mp4)
- predicted masks: [results/masks/part1/bmx-trees](results/masks/part1/bmx-trees)
- visualization assets: [results/visualizations/part1/bmx-trees](results/visualizations/part1/bmx-trees)
- metric CSVs: [results/metrics/iou_results.csv](results/metrics/iou_results.csv) and [results/metrics/psnr_ssim.csv](results/metrics/psnr_ssim.csv)

## Notes on Metrics

IoU is computed when reference masks are available. For `bmx-trees` and `tennis`, the evaluator automatically uses the dataset-specific mask folders defined in [configs/common.yaml](configs/common.yaml).

PSNR and SSIM currently default to background-only evaluation when no clean restored target frames are available. That means the metric is computed on pixels outside the union of predicted and reference foreground masks, which makes it suitable for background preservation analysis in object removal settings.

## Next Stages

Part 2 and Part 3 entrypoints already exist, but only Part 1 is fully implemented at this stage. The next major milestones are:

- higher-quality SOTA mask generation and inpainting for Part 2
- exploration and failure-case improvements for Part 3
- report-oriented summary tables and final packaging scripts

## Part 2 Progress

Part 2 is now implemented beyond the initial interface scaffold. The current Part 2 workflow includes:

- SAM2-based mask generation via sequence-specific `box`/`points` prompts in [configs/part2_sota.yaml](configs/part2_sota.yaml)
- ProPainter video inpainting to produce coherent restored output
- saved mask outputs separated into `objects/` and `combined/` for multi-object scenes
- metrics evaluation updated to save results under `results/metrics/part2/<sequence>/`

Current Part 2 outputs follow this structure:

- `results/masks/part2/<sequence>/objects/`
- `results/masks/part2/<sequence>/combined/`
- `results/videos/part2/<sequence>_inpainted.mp4`
- `results/metrics/part2/<sequence>/iou_results.csv`
- `results/metrics/part2/<sequence>/psnr_ssim.csv`

The evaluator now uses the `combined/` masks by default for overall foreground IoU when Part 2 produces multiple object masks.

## Part 2 Run Commands

### 1. Standard Part 2 Run

```bash
conda activate project3
python src/part2_sota/pipeline_part2.py \
	--sequence datasets \
	--common-config configs/common.yaml \
	--phase-config configs/part2_sota.yaml
```

### Example: Run Part 2 on a Specific Sequence

```bash
conda activate project3
python src/part2_sota/pipeline_part2.py \
	--sequence bmx-trees \
	--common-config configs/common.yaml \
	--phase-config configs/part2_sota.yaml
```

```bash
conda activate project3
python src/part2_sota/pipeline_part2.py \
	--sequence tennis \
	--common-config configs/common.yaml \
	--phase-config configs/part2_sota.yaml
```

### 2. Evaluate Part 2 Results

```bash
conda activate project3
python scripts/evaluate_metrics.py \
	--phase-config configs/part2_sota.yaml \
	--sequence datasets 
```

### Example: Evaluate Part 2 on a Specific Sequence

```bash
conda activate project3
python scripts/evaluate_metrics.py \
	--phase-config configs/part2_sota.yaml \
	--sequence bmx-trees
```

```bash
conda activate project3
python scripts/evaluate_metrics.py \
	--phase-config configs/part2_sota.yaml \
	--sequence tennis
```

The evaluator also auto-detects the Part 2 video under `results/videos/part2/` when the standard `*_inpainted.mp4` naming pattern is used.
