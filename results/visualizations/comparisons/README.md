# 横向对比可视化文本

本目录用于组织 Part 1 / Part 2 / Part 3 的横向对比说明，目标是把已有产物直接映射为报告中的 comparison figures 和 result tables。

## 对比对象

建议统一比较三条 pipeline：

| 方法 | 分割来源 | 修复来源 | 主要优势 | 主要风险 |
| --- | --- | --- | --- | --- |
| Part 1 Baseline | YOLOv8-Seg + 光流动态筛选 | 传统时序传播 + `cv2.inpaint` | 可解释、依赖轻、实现透明 | 易模糊、纹理重建弱 |
| Part 2 SOTA | SAM2 prompt video masks | ProPainter | 掩码质量与修复质量显著提升 | 对 prompt 质量敏感 |
| Part 3 Exploration | Part 2 coarse mask + SAM3 refinement | ProPainter | 边界更稳、候选选择更细 | 依赖 checkpoint，可触发 coarse fallback |

## 推荐 Figure 1: 同帧三阶段对比

建议对同一 `sequence`、同一 `frame_xxxx.png` 做三行或三列对比：

- Part 1: `results/visualizations/part1/{sequence}/comparisons/frame_xxxx.png`
- Part 2: `results/visualizations/part2/{sequence}/comparisons/frame_xxxx.png`
- Part 3: `results/visualizations/part3/{sequence}/comparisons/frame_xxxx.png`

三联图内部已经包含：

- Original
- Mask Overlay
- Restored

因此横向比较时不需要再额外拼接原始帧，只需要保证三阶段选择同一 frame index。

建议 caption：

Figure X compares the three project stages on the same frame. Part 1 removes the target with an interpretable but weaker classical pipeline, Part 2 improves mask completeness and texture coherence using SAM2 plus ProPainter, and Part 3 further refines boundaries by using SAM3 on top of the Part 2 coarse masks.

## 推荐 Figure 2: 掩码质量对比

建议从 `mask_overlays/` 提取同一帧，重点展示：

- Part 1 是否误保留静态区域，或遗漏高速目标边缘
- Part 2 是否覆盖完整动态目标与影子
- Part 3 是否进一步改善边界贴合度，减少粗掩码泄漏

对应数据源：

- `results/visualizations/part1/{sequence}/mask_overlays/`
- `results/visualizations/part2/{sequence}/mask_overlays/`
- `results/visualizations/part3/{sequence}/mask_overlays/`

建议正文描述模板：

The mask overlays show a clear progression from heuristic motion filtering to prompt-driven segmentation and finally to coarse-to-fine refinement. Part 1 emphasizes motion explainability, Part 2 improves object completeness, and Part 3 mainly improves boundary precision and mask stability.

## 推荐 Figure 3: 解释性可视化对比

解释图并不是三阶段完全同构，因此应按“解释目标”而不是按“像素外观”比较：

| 阶段 | 推荐子图 | 解释重点 |
| --- | --- | --- |
| Part 1 | `motion_scores/` | 光流分数、动态/静态筛选依据 |
| Part 2 | `motion_scores/` | SAM2 追踪到的对象实例与 prompt 结果 |
| Part 3 | `motion_scores/` | SAM3 候选 mask 的得分与一致性筛选 |

这张图适合放在 Method 的后半部分或 Experiments 的可解释性小节，而不是主 quantitative comparison 图。

## 推荐 Table 1: 定量结果总表

按 `sequence` 汇总三阶段的指标文件：

- `results/metrics/part1/{sequence}/iou_results.csv`
- `results/metrics/part1/{sequence}/psnr_ssim.csv`
- `results/metrics/part2/{sequence}/iou_results.csv`
- `results/metrics/part2/{sequence}/psnr_ssim.csv`
- `results/metrics/part3/{sequence}/iou_results.csv`
- `results/metrics/part3/{sequence}/psnr_ssim.csv`

建议表头：

| Sequence | Method | IoU mean | IoU recall | PSNR | SSIM | Notes |
| --- | --- | --- | --- | --- | --- | --- |

建议说明：

- `bmx-trees` 和 `tennis` 更适合 quantitative comparison，因为已有参考 mask。
- `wild_video_frames` 更适合 qualitative comparison，因为没有标准 GT。
- 若 PSNR / SSIM 采用 background-only evaluation，需要在 caption 中明确说明。

## 当前仓库中可直接做对比的素材

从现有目录看，以下序列已经具备完整三阶段可视化结构：

- `bmx-trees`
- `tennis`
- `wild_video_frames`

其中当前会话已确认 `wild_video_frames` 的三阶段 pipeline 均已成功运行，因此可以优先将它作为主 qualitative comparison 序列。

## 推荐版式

1. 一张主图对比三阶段的 `comparisons/frame_xxxx.png`。
2. 一张辅图对比三阶段的 `mask_overlays/frame_xxxx.png`。
3. 一张解释图展示 `motion_scores/frame_xxxx.png`。
4. 一张表汇总 `bmx-trees` 与 `tennis` 的 IoU / PSNR / SSIM。

这样能与 `Overall_Plan.md` 中“量化 + 定性 + flowchart”三类要求严格对齐。