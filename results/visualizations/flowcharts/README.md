# 可视化流程图文本

本目录用于存放报告中 Method / Pipeline flowchart 的文字底稿。内容同时对齐两部分来源：

- `Overall_Plan.md` 中对 Part 1 / Part 2 / Part 3 的任务拆解
- `src/` 中已经实现并跑通的真实流程

当前代码里的实际输出路径为：

- `results/visualizations/part1/{sequence_name}/`
- `results/visualizations/part2/{sequence_name}/`
- `results/visualizations/part3/{sequence_name}/`

其中每个 phase 都会导出三类图像：

- `motion_scores/`: 候选目标或候选掩码解释图
- `mask_overlays/`: 最终 removal mask 叠加图
- `comparisons/`: original / mask overlay / restored 三联图

## 总体方法流程图

```mermaid
flowchart LR
    A[Input video frames] --> B[Part 1 Baseline]
    A --> C[Part 2 SOTA]
    A --> D[Part 3 Exploration]

    B --> B1[YOLOv8-Seg candidate masks]
    B1 --> B2[Lucas-Kanade motion filtering]
    B2 --> B3[Mask merge and temporal postprocess]
    B3 --> B4[Traditional temporal propagation + cv2.inpaint fallback]
    B4 --> B5[Part 1 masks, video, explainability images]

    C --> C1[SAM2 prompted video masks]
    C1 --> C2[Object masks + combined mask export]
    C2 --> C3[Mask cleanup]
    C3 --> C4[ProPainter video inpainting]
    C4 --> C5[Part 2 masks, video, report images]

    D --> D1[Reuse Part 2 coarse masks]
    D1 --> D2[SAM3 candidate refinement]
    D2 --> D3[Consistency gate or coarse-mask fallback]
    D3 --> D4[Postprocess refined masks]
    D4 --> D5[ProPainter video inpainting]
    D5 --> D6[Part 3 masks, video, report images]
```

## Part 1: Baseline 流程图文本

对应实现：

- `src/part1_baseline/pipeline_part1.py`
- `src/part1_baseline/mask_extraction_yolo.py`
- `src/part1_baseline/dynamic_judgment.py`
- `src/part1_baseline/inpaint_traditional.py`

```mermaid
flowchart TD
    A[Frame sequence] --> B[MaskExtractorYOLO]
    B --> C[Candidate instance masks]
    C --> D[DynamicObjectJudge]
    D --> E[estimate_mask_motion with sparse LK optical flow]
    E --> F{Motion above threshold?}
    F -->|Yes| G[Keep dynamic instance]
    F -->|No| H[Reject static instance]
    G --> I[Merge instance masks]
    H --> I
    I --> J[postprocess_mask]
    J --> K[Save binary mask]
    J --> L[TraditionalVideoInpainter]
    L --> M[Temporal propagation]
    L --> N[cv2.inpaint spatial fallback]
    M --> O[Restored frame sequence]
    N --> O
    O --> P[Export video]
    O --> Q[Export comparison triptychs]
    C --> R[Export motion score overlays]
    J --> S[Export removal mask overlays]
```

建议图注：

Part 1 以 YOLOv8-Seg 生成候选实例，再用稀疏 Lucas-Kanade 光流判断目标是否真正运动，最后通过传统时序传播与 OpenCV inpainting 进行修复。该流程可解释性强，但在大遮挡和复杂纹理区域容易出现模糊与结构断裂。

## Part 2: SOTA 流程图文本

对应实现：

- `src/part2_sota/pipeline_part2.py`
- `src/part2_sota/mask_sam2.py`
- `src/part2_sota/inpaint_pro_painter.py`
- `configs/part2_sota.yaml`

```mermaid
flowchart TD
    A[Frame sequence] --> B[Load SAM2 prompts]
    B --> C[SAM2MaskGenerator.generate]
    C --> D[Per-object masks across video]
    D --> E[Export objects/ masks]
    D --> F[Union to combined mask]
    F --> G[postprocess_mask]
    G --> H[Export combined masks]
    H --> I[ProPainterInpainter.inpaint]
    I --> J[Restored video]
    D --> K[Render tracked-object overlays]
    G --> L[Render final mask overlays]
    J --> M[Render original / mask / restored comparisons]
```

建议图注：

Part 2 使用基于 prompt 的 SAM2 进行视频目标分割，再将多目标掩码融合为单一 removal mask，输入 ProPainter 进行时序一致的视频修复。与 Part 1 相比，该阶段显著提升了遮挡区域的纹理连续性和整体视觉质量。

## Part 3: Exploration 流程图文本

对应实现：

- `src/part3_exploration/pipeline_part3.py`
- `src/part3_exploration/sam3_upgrade.py`
- `configs/part3_exploration.yaml`

```mermaid
flowchart TD
    A[Frame sequence] --> B[Load Part 2 coarse masks]
    B --> C{Use combined mask or objects union?}
    C --> D[SAM3UpgradeRefiner]
    D --> E[Convert coarse mask to bounding box prompt]
    E --> F[Generate SAM3 candidate masks]
    F --> G[Compute IoU, precision, recall, area ratio]
    G --> H{Pass consistency gate?}
    H -->|Yes| I[Use refined SAM3 mask]
    H -->|No| J[Fallback to coarse mask]
    I --> K[Optional postprocess_mask]
    J --> K
    K --> L[Export refined masks]
    L --> M[ProPainterInpainter.inpaint]
    M --> N[Export video and comparison images]
    F --> O[Export candidate overlay visualizations]
```

建议图注：

Part 3 将 Part 2 生成的 coarse mask 作为先验，通过 SAM3 对边界进行再细化，并使用一致性门控防止 refinement 过度偏离原始目标区域。若候选掩码不满足约束，则回退到 coarse mask，以保证鲁棒性。

## 报告中推荐的流程图摆放方式

1. Method 总图使用“总体方法流程图”。
2. Part 1 / Part 2 / Part 3 各自的小图可放在 Method 或 Appendix 中。
3. 若版面有限，正文保留一张总图，再在 caption 中概括三阶段差异：

   - Part 1: YOLO + optical flow + traditional inpainting
   - Part 2: SAM2 + ProPainter
   - Part 3: Part 2 coarse mask + SAM3 refinement + ProPainter