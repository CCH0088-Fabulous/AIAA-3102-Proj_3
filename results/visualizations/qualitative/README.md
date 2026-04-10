# 定性展示可视化文本

本目录用于整理报告里的 qualitative results 叙事文本，包括选帧规则、展示重点、caption 模板，以及当前仓库中已经存在的结果素材映射。

## 定性展示目标

根据 `Overall_Plan.md`，定性分析需要回答三类问题：

1. 动态目标是否被完整移除。
2. 背景纹理是否自然、连续且时序一致。
3. 失败案例是否被清楚展示，并说明 Part 3 为何能改进或为何仍然失败。

## 建议展示序列

优先级建议如下：

1. `wild_video_frames`: 作为主 qualitative case study
2. `bmx-trees`: 作为标准 benchmark 示例
3. `tennis`: 作为多目标、细长结构与影子场景的困难案例

原因：

- `wild_video_frames` 已经在当前仓库中完成 Part 1 / Part 2 / Part 3 全流程运行。
- `bmx-trees` 和 `tennis` 同时适合 qualitative 与 quantitative 展示。
- `tennis` 含人、球拍、球、阴影，最能体现 mask quality 的差异。

## 建议选帧规则

不要随机截帧。每个序列至少选 4 到 6 帧，并覆盖以下时刻：

- 目标刚进入场景
- 遮挡面积最大的时刻
- 目标与背景边界最复杂的时刻
- 阴影或细长结构最明显的时刻
- 目标离开后背景完全暴露的时刻

如果暂时没有人工筛选时间，可先用均匀采样作为默认方案。当前 `wild_video_frames` 的 comparison 输出已经覆盖 `frame_0000.png` 到 `frame_0136.png`，因此默认可选：

- `frame_0000.png`
- `frame_0027.png`
- `frame_0054.png`
- `frame_0081.png`
- `frame_0108.png`
- `frame_0136.png`

这组帧适合先搭建 Appendix 或草稿版对比图，再根据视觉效果替换为事件驱动的关键帧。

## 当前可直接引用的素材路径

### Wild Video

- Part 1 对比图：`results/visualizations/part1/wild_video_frames/comparisons/`
- Part 2 对比图：`results/visualizations/part2/wild_video_frames/comparisons/`
- Part 3 对比图：`results/visualizations/part3/wild_video_frames/comparisons/`
- Part 1 视频：`results/videos/part1/wild_video_frames_part1.mp4`
- Part 2 视频：`results/videos/part2/wild_video_frames_inpainted.mp4`
- Part 3 视频：`results/videos/part3/wild_video_frames_part3.mp4`

### Benchmark Sequences

- `results/visualizations/part1/bmx-trees/`
- `results/visualizations/part2/bmx-trees/`
- `results/visualizations/part3/bmx-trees/`
- `results/visualizations/part1/tennis/`
- `results/visualizations/part2/tennis/`
- `results/visualizations/part3/tennis/`

## 每阶段的定性观察重点

### Part 1

建议强调：

- 光流筛选逻辑清晰，容易解释为什么某个目标被保留或过滤。
- 当背景纹理复杂或遮挡面积较大时，传统 inpainting 容易产生模糊、拖影或纹理断裂。
- 适合作为 baseline，而不是最终视觉质量最优方案。

可直接配合：

- `motion_scores/` 展示筛选依据
- `comparisons/` 展示恢复质量局限

### Part 2

建议强调：

- SAM2 让目标覆盖更完整，尤其适合连续视频中的主目标跟踪。
- ProPainter 显著改善了背景纹理延续与时序稳定性。
- 但结果仍受 prompt 设计质量影响，复杂场景下可能出现边界偏粗或局部漏分割。

### Part 3

建议强调：

- Part 3 的核心不是替换整个 pipeline，而是在 Part 2 coarse masks 之上做 refinement。
- SAM3 候选筛选引入 IoU、precision、recall、area ratio 等一致性约束，目标是提高边界贴合度，同时避免 refinement 漂移。
- 若 refinement 候选不可靠，代码会回退到 coarse mask，因此定性结果通常更稳，而不是更激进。

## 失败案例写法模板

建议至少保留一个 failure case 面板，并使用下面的结构：

Failure case X shows that Part 1 fails to reconstruct fine background texture after large-object removal, while Part 2 improves temporal coherence but still keeps a coarse boundary near the moving silhouette. Part 3 reduces the boundary leakage by refining the coarse mask with SAM3, although minor artifacts remain in heavily occluded regions.

如果 Part 3 没有明显优于 Part 2，也应如实写：

In this case, Part 3 falls back to the coarse Part 2 mask because the SAM3 candidates do not satisfy the consistency gate. This avoids catastrophic drift but also limits the visible improvement.

## 图注模板

### 主定性结果图

Figure X presents qualitative comparisons on the same video frame across the three project stages. Part 1 provides an interpretable classical baseline, Part 2 improves object coverage and inpainting realism with SAM2 plus ProPainter, and Part 3 further refines the coarse mask with SAM3 to obtain cleaner boundaries.

### 失败案例图

Figure Y highlights a challenging case with thin structures, shadows, or large occlusions. The baseline method leaves visible artifacts, the SOTA pipeline restores most of the background, and the exploration stage mainly improves boundary precision or robustness through refinement and fallback control.

## 版面组织建议

1. 正文放 1 到 2 组主 qualitative comparisons。
2. 每组包含同一帧的 Part 1 / Part 2 / Part 3 对比。
3. Appendix 再补充 mask overlays 与 motion/object candidate overlays。
4. 至少保留 1 组 failure case，避免报告只展示成功样例。

这样能与项目要求中的“clean and aesthetic”展示目标一致，也能和本仓库当前的真实输出结构完全对齐。