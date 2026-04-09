# A Staged Video Object Removal and Inpainting System

## Note on Scope

This report is written from the current repository state, using the project README, configuration files, shared modules in `src/common`, stage-specific pipelines, and the integrated model repositories under `models/`. It intentionally does **not** inspect generated artifacts under `results/visualizations/` or any visualization folder. The emphasis is therefore on architecture, implementation, current integration status, and engineering readiness rather than on visual case studies.

## 1. Introduction

This project implements a staged system for video object removal and background restoration. The main application is the removal of dynamic foreground objects such as people, bicycles, rackets, or balls from short video sequences, followed by the reconstruction of visually plausible backgrounds. Rather than relying on a single monolithic design, the repository is organized as a progressive three-stage pipeline:

1. A classical baseline that combines segmentation, motion filtering, mask cleanup, temporal background borrowing, and traditional image inpainting.
2. A stronger state-of-the-art pipeline that replaces the baseline mask generator with SAM2 and the baseline restoration module with ProPainter.
3. An exploration stage that refines coarse masks with SAM3 before reusing the stronger inpainting backend.

This staged organization is useful for both engineering and research. It preserves an interpretable baseline, introduces a clearly improved modern pipeline, and keeps an experimental branch for model upgrades without destabilizing earlier phases.

## 2. Problem Setting and Design Goals

The problem addressed by the repository is video object removal under realistic data and implementation constraints. A successful system must identify the target object, produce temporally stable masks, avoid deleting static instances that should remain in the scene, and restore missing regions without introducing obvious artifacts.

From the current codebase, the design goals can be summarized as follows:

- Support both named datasets and direct frame folders.
- Keep stage entrypoints consistent across phases.
- Separate shared infrastructure from stage-specific logic.
- Allow classical and modern approaches to coexist in the same repository.
- Produce masks, restored videos, and quantitative evaluation outputs in a reproducible directory structure.
- Maintain enough visualization hooks for debugging, while keeping the core method independent from those visualizations.

## 3. Repository-Level Architecture

The repository is structured around configurations, shared utilities, stage-specific pipelines, third-party model repositories, and output folders.

| Layer | Main Location | Role |
| --- | --- | --- |
| Shared configuration | `configs/common.yaml` | Dataset aliases, naming rules, output roots, video format conventions |
| Stage configuration | `configs/part1_baseline.yaml`, `configs/part2_sota.yaml`, `configs/part3_exploration.yaml` | Per-phase models, thresholds, output directories, and pipeline switches |
| Shared utilities | `src/common/` | Configuration loading, dataset resolution, mask processing, metrics, optical flow, visualization helpers |
| Stage 1 | `src/part1_baseline/` | Baseline object removal and classical inpainting |
| Stage 2 | `src/part2_sota/` | SAM2-based video segmentation and ProPainter-based restoration |
| Stage 3 | `src/part3_exploration/` | SAM3-based refinement and future diffusion-oriented expansion |
| External model repos | `models/` | SAM2, SAM3, ProPainter, and YOLO assets |
| Evaluation scripts | `scripts/evaluate_metrics.py` | Unified IoU, PSNR, and SSIM evaluation |

The resulting design is cleanly modular. Shared logic is centralized in `src/common`, while each stage remains independent enough to be executed from its own entrypoint.

## 4. Shared Infrastructure in `src/common`

### 4.1 Configuration and Dataset Resolution

The shared configuration layer is implemented primarily in `src/common/config.py`. It provides the following capabilities:

- YAML loading through a common helper.
- Resolution of dataset aliases such as `bmx_trees` to the canonical key `bmx-trees`.
- Frame directory discovery for configured datasets and direct folder inputs.
- Automatic fallback for nested dataset folders.
- Unified naming for frame, mask, and video outputs.
- Creation of per-phase output directories.

This design matters because the same dataset can be referenced in multiple ways during development. The repository normalizes these references early, which reduces brittle path handling inside the actual pipelines.

At the dataset level, `configs/common.yaml` currently defines canonical support for:

- `bmx-trees`
- `tennis`
- `davis` sequences resolved from DAVIS roots

The system also supports direct frame-folder execution when the user supplies a folder instead of a dataset key.

### 4.2 Binary Mask Processing

The file `src/common/mask_utils.py` implements a compact but important mask postprocessing stack. The available operations are:

- binary normalization
- dilation
- flood-fill-based hole filling
- connected-component filtering by minimum area
- short-window temporal voting

This is a strong engineering choice. The segmentation backend can change across stages, but the repository keeps a consistent postprocessing contract that stabilizes masks before restoration. In practical terms, this improves contour coverage, suppresses isolated noise, and reduces frame-to-frame flicker.

### 4.3 Sparse Optical Flow Utilities

The motion subsystem in `src/common/optical_flow.py` is based on sparse Lucas-Kanade tracking. The implementation includes:

- grayscale conversion and mask normalization
- Shi-Tomasi feature extraction inside candidate masks
- pyramidal Lucas-Kanade point tracking
- backward-tracking consistency checks
- motion summary statistics including mean, median, and maximum displacement

This module is central to Stage 1 because it decides whether a detected object is actually moving. That prevents the baseline from removing every segmented object indiscriminately.

### 4.4 Quantitative Metrics

The file `src/common/metrics.py` provides the repository-wide evaluation core. The implemented measures are:

- IoU for binary mask agreement
- PSNR for image fidelity
- SSIM for structural similarity
- a background-valid-mask constructor that excludes the union of foreground masks during background-preservation evaluation

This is a particularly sensible design for object removal. In many cases, there is no clean ground-truth video showing the same scene without the target object. The repository addresses that by evaluating restoration quality only on background regions that should remain unchanged.

### 4.5 Visualization Interfaces

The file `src/common/visualization.py` generates three classes of debugging outputs:

- motion-score overlays
- mask overlays
- before-mask-restored comparison panels

Although these interfaces are part of the implementation, this report does not inspect the actual exported visualization folders. Their significance here is architectural: the project is designed to expose intermediate reasoning for debugging and reporting without mixing visualization logic into the restoration logic itself.

## 5. Stage 1: Classical Baseline Pipeline

### 5.1 Pipeline Logic

The Stage 1 entrypoint is `src/part1_baseline/pipeline_part1.py`. The implemented workflow is:

1. Resolve the input sequence.
2. Run YOLOv8-Seg instance segmentation on each frame.
3. Filter candidate instances with sparse optical-flow motion analysis.
4. Merge the selected instance masks into a frame-level removal mask.
5. Postprocess the merged mask.
6. Restore the missing region by temporal borrowing from nearby frames.
7. Fill unresolved holes with OpenCV inpainting.
8. Save masks, video, and optional debugging artifacts.

This is no longer a toy detection demo. It is a full classical object-removal pipeline.

### 5.2 YOLOv8-Seg for Candidate Extraction

The detector wrapper is implemented in `src/part1_baseline/mask_extraction_yolo.py`. It uses the Ultralytics YOLO interface and loads the segmentation checkpoint configured in `configs/part1_baseline.yaml`, which defaults to `models/yolo_v8_seg/yolov8x-seg.pt`.

The current default target classes are:

- class `0`: person
- class `1`: bicycle

Each valid instance mask is resized if necessary to match the frame size and is converted into a strict binary mask.

### 5.3 Dynamic Object Judgment

The motion filtering logic is implemented in `src/part1_baseline/dynamic_judgment.py`. For every candidate mask, the system estimates sparse motion between the previous and current frame and then aggregates the motion magnitude. By default, the pipeline uses:

- median motion aggregation
- motion threshold of `1.5`
- minimum tracked points of `8`
- keep-if-undetermined behavior when insufficient points are available

This is a strong baseline strategy. It reduces false removals of static people or bicycles while remaining much cheaper than dense video segmentation or tracking-by-foundation-model approaches.

### 5.4 Traditional Restoration Module

The restoration backend is implemented in `src/part1_baseline/inpaint_traditional.py`. It has two parts:

- temporal borrowing from neighboring frames at the same pixel locations when those locations are unmasked in the candidate frame
- spatial fallback using OpenCV `cv2.inpaint`

The currently supported fallback modes are Telea and Navier-Stokes. The default configuration uses Telea with an inpainting radius of `3.0` and a temporal window of `3`.

This is an appropriate baseline because it directly matches the project objective of combining simple temporal reasoning with a traditional image completion fallback.

### 5.5 Strengths and Limitations of Stage 1

Strengths:

- interpretable end-to-end logic
- relatively lightweight dependencies compared with large video foundation models
- explicit motion reasoning rather than blind foreground deletion
- useful for ablations and sanity checks

Limitations:

- segmentation quality is tied to a generic object detector
- sparse motion can fail on low-texture objects or weak inter-frame displacement
- temporal borrowing assumes usable clean pixels exist nearby in time
- OpenCV fallback is local and cannot synthesize complex texture with long-range consistency

## 6. Stage 2: SAM2 + ProPainter SOTA Pipeline

### 6.1 Overall Role of Stage 2

Stage 2 replaces both the mask-generation quality bottleneck and the restoration quality bottleneck from Stage 1. The entrypoint is `src/part2_sota/pipeline_part2.py`, and the phase configuration is `configs/part2_sota.yaml`.

The high-level flow is:

1. Resolve a sequence and prompt specification.
2. Use SAM2 to generate object masks across the video.
3. Merge and lightly postprocess the masks.
4. Save object-wise and combined masks.
5. Pass the combined masks to ProPainter.
6. Export the restored video and optional overlays.

### 6.2 SAM2 Integration

The SAM2 wrapper is implemented in `src/part2_sota/mask_sam2.py`. The integration is careful and practical in several ways:

- it adds the local SAM2 repository to `sys.path`
- it initializes SAM2 through Hydra configuration
- it searches for available checkpoints under `models/sam2/checkpoints/`
- it supports both point prompts and box prompts
- it propagates prompts through the full video using the SAM2 video predictor
- it converts PNG frame folders to temporary JPEG folders when needed for compatibility

The wrapper searches for the following checkpoint family in order:

- `sam2.1_hiera_large.pt`
- `sam2.1_hiera_base_plus.pt`
- `sam2.1_hiera_small.pt`
- `sam2.1_hiera_tiny.pt`

This aligns with the official SAM2 repository, where SAM2 is a promptable image and video segmentation foundation model with streaming-memory video support.

### 6.3 Prompt Engineering and Sequence Handling

Stage 2 supports prompts from two sources:

- command-line prompt arguments
- sequence-specific prompt presets inside `configs/part2_sota.yaml`

The repository already includes custom prompt logic for at least:

- `bmx-trees`, where a box captures the person and bicycle jointly
- `tennis`, where multiple prompted objects and shadows are considered

This is a pragmatic compromise between fully automatic segmentation and manual annotation. It keeps the system controllable while exploiting the much stronger segmentation capacity of a modern foundation model.

### 6.4 ProPainter Integration

The restoration backend is implemented in `src/part2_sota/inpaint_pro_painter.py`. This module integrates the official ProPainter repository and uses three core components:

- RAFT-based bidirectional flow estimation
- recurrent flow completion
- the ProPainter inpainting generator

The wrapper automatically downloads pretrained weights into `models/ProPainter/weights/` on first use when they are not already present. It also adds a practical memory-aware resize policy that reduces processing resolution for longer videos before resizing results back to the original frame size.

This is an important engineering improvement. Video inpainting quality is not only about the inpainting network itself; it also depends on whether the model can run reliably on the available GPU without running out of memory.

### 6.5 Stage 2 Mask Export Strategy

Stage 2 saves two mask products per sequence:

- object masks in an `objects/` subdirectory
- merged masks in a `combined/` subdirectory

The combined-mask export is important because the inpainting backend only needs a single binary region per frame, while object-wise masks remain valuable for debugging and possible future object-level reasoning.

### 6.6 Strengths and Limitations of Stage 2

Strengths:

- much stronger video-aware segmentation than the Stage 1 detector
- far better temporal restoration quality than classical inpainting
- flexible prompt interface for difficult sequences
- improved memory handling for longer videos

Limitations:

- dependence on SAM2 checkpoints and ProPainter dependencies
- prompt quality still matters for difficult scenes
- higher compute and memory cost than the baseline
- more fragile environment setup because several external repositories must remain compatible

## 7. Stage 3: Exploratory SAM3 Refinement Pipeline

### 7.1 Motivation

Stage 3 is designed as an upgrade path rather than a full replacement of Stage 2. The idea is to start from coarse masks already generated by Stage 2, refine them with a stronger segmentation foundation model, and then reuse the stable ProPainter backend for restoration.

The Stage 3 entrypoint is `src/part3_exploration/pipeline_part3.py`, and its main refinement logic is in `src/part3_exploration/sam3_upgrade.py`.

### 7.2 Coarse-to-Refined Mask Strategy

The pipeline first loads baseline masks from Stage 2, either:

- directly from combined masks, or
- by taking the union of object-wise masks

It then uses those coarse masks as prompts for SAM3 image-level refinement. This is a sensible design because it treats Stage 2 as a proposal generator and Stage 3 as a refinement layer rather than forcing SAM3 to solve the full problem from scratch.

### 7.3 SAM3-Based Refinement Logic

The `SAM3UpgradeRefiner` performs several nontrivial operations:

- resolve or download a SAM3 checkpoint
- build a SAM3 image model and processor
- convert the coarse mask into a bounding box prompt
- expand the box slightly using a configurable ratio
- run SAM3 to generate candidate masks
- score candidates against the coarse mask using overlap-based statistics
- accept a refined mask only when consistency gates are satisfied
- fall back to the coarse mask when the refinement is not trustworthy

The gating criteria include:

- coarse-mask IoU
- area-ratio consistency
- candidate precision with respect to the coarse mask
- coarse-mask recall under the refined candidate

This is a thoughtful engineering safeguard. It prevents a stronger but more open-ended model from drifting away from the intended object region.

### 7.4 Relation to the Official SAM3 Repository

The local `models/sam3` repository describes SAM3 as a promptable segmentation foundation model for images and videos with richer concept-level capabilities than SAM2. In this project, however, the integration uses the image model and processor to refine already available per-frame mask proposals. That is an appropriate first step because it limits complexity while still testing whether the newer model improves mask precision.

### 7.5 Current Practical Constraints

Stage 3 is the most environment-sensitive branch in the repository. Based on the current implementation and validated repository notes, the main constraints are:

- the `sam3` package must be installed in the environment
- `pycocotools` is required at runtime
- checkpoint availability is critical
- automatic checkpoint download can fail when Hugging Face access or authentication is unavailable
- the upstream SAM3 dependency stack can conflict with the broader project environment

Therefore, Stage 3 should be considered a concrete experimental implementation, but not yet the most deployment-ready path in the repository.

### 7.6 Diffusion Branch Status

The file `src/part3_exploration/diffusion_controlnet.py` currently defines only a placeholder `ControlNetInpainter` interface that raises `NotImplementedError`. This means the diffusion-oriented branch is planned and structurally reserved, but it is not part of the current executable system.

## 8. Environment and Dependency Stack

The top-level `requirements.txt` indicates a GPU-oriented environment built around:

- PyTorch with CUDA 12.8 support
- TorchVision and TorchAudio
- OpenCV
- SciPy
- PyYAML
- Ultralytics
- Matplotlib
- NumPy

This top-level environment is sufficient for the baseline and much of the project infrastructure, but the model repositories impose extra requirements:

- SAM2 expects its repository to be installed and a valid checkpoint to be present locally.
- ProPainter requires its own Python dependencies and downloads its pretrained weights on first use.
- SAM3 introduces the heaviest compatibility burden because its preferred environment can diverge from the main project environment.

From a software-engineering perspective, the repository is already at the point where environment isolation by model family would be justified if the project continues to grow.

## 9. Evaluation Protocol

The repository uses `scripts/evaluate_metrics.py` as a unified evaluator for all phases. Its evaluation policy is well matched to the object-removal task.

### 9.1 Mask Evaluation

When reference masks are available, the script computes IoU between predicted masks and ground-truth masks. This applies directly to sequences that provide annotation masks or to DAVIS sequences resolved through the configured annotation root.

### 9.2 Restoration Evaluation

For restored videos, the script supports two modes:

- `full_reference`, when clean target frames are explicitly provided
- `background_preservation`, when no clean reference video exists

The second mode is especially important in this project. It evaluates only the background area outside the union of foreground masks, which is exactly the region that should remain visually stable after object removal.

### 9.3 Output Convention

Metrics are written per phase and per sequence, with per-frame rows and a mean summary row. This makes the evaluator suitable both for debugging and for later report generation.

## 10. Current Build Status

At the present stage, the repository can be characterized as follows.

| Stage | Core Idea | Status | Practical Readiness |
| --- | --- | --- | --- |
| Stage 1 | YOLOv8-Seg + LK motion filtering + temporal borrowing + OpenCV inpainting | Implemented | Stable baseline |
| Stage 2 | SAM2 prompting and propagation + ProPainter restoration | Implemented | Strong primary pipeline |
| Stage 3 | SAM3 refinement over Stage 2 masks + ProPainter reuse | Implemented experimentally | Conditional on environment and checkpoints |
| Diffusion branch | ControlNet-based inpainting refinement | Reserved only | Not implemented |

The most important conclusion is that the repository is already beyond an initial skeleton. It contains two complete end-to-end object-removal pipelines and one meaningful exploratory branch. The baseline is interpretable and reproducible, the SOTA branch is the main high-quality path, and the SAM3 branch is a legitimate research extension with clear external dependencies.

## 11. Main Technical Contributions of the Current Repository

From the codebase as it stands today, the most important technical contributions are the following:

- a unified multi-stage architecture for video object removal
- consistent configuration and dataset resolution across phases
- explicit dynamic-object filtering in the classical baseline
- practical integration of SAM2 for prompted video mask propagation
- practical integration of ProPainter with adaptive preprocessing for runtime stability
- coarse-to-refined SAM3 mask refinement with conservative acceptance gates
- a shared evaluation framework that supports both mask accuracy and restoration quality

These contributions make the repository useful not only as a project submission artifact, but also as a compact experimental platform for comparing classical and modern removal strategies.

## 12. Limitations and Recommended Next Steps

The current system is technically solid, but several limitations are clear from the codebase:

- Stage 2 still depends on prompt design rather than fully automatic target discovery.
- Stage 3 depends on external checkpoint access and a less stable environment configuration.
- The diffusion branch is not yet implemented.
- The evaluation pipeline is present, but a formal comparison table across all supported datasets is not yet centralized in a document.

The most rational next steps would be:

1. stabilize the SAM3 execution environment and checkpoint management
2. finish the diffusion-oriented ControlNet branch only after Stage 3 becomes reproducible
3. add a single benchmark summary table across `bmx-trees`, `tennis`, DAVIS, and custom wild-video sequences
4. formalize prompt presets or semi-automatic prompt generation for Stage 2 and Stage 3

## 13. Conclusion

This repository already represents a well-structured staged system for video object removal and inpainting. Stage 1 provides an interpretable classical baseline built from YOLOv8 segmentation, sparse optical-flow reasoning, mask postprocessing, temporal background borrowing, and OpenCV inpainting. Stage 2 upgrades both segmentation and restoration through SAM2 and ProPainter, making it the strongest production candidate in the current codebase. Stage 3 extends the system in a research-oriented direction by refining coarse masks with SAM3 while reusing the established ProPainter backend.

The overall architecture is coherent, modular, and technically defensible. The shared infrastructure in `src/common` is a major strength because it keeps configuration, metrics, optical flow, and mask processing consistent across phases. The project is therefore already in a strong phase of staged construction: it has a valid baseline, a high-quality modern pipeline, and a concrete experimental path for further improvement.

## References

1. Ultralytics YOLOv8 segmentation framework.
2. Meta AI, SAM 2: Segment Anything in Images and Videos.
3. Meta Superintelligence Labs, SAM 3: Segment Anything with Concepts.
4. Shangchen Zhou et al., ProPainter: Improving Propagation and Transformer for Video Inpainting, ICCV 2023.
5. OpenCV inpainting methods: Telea and Navier-Stokes.