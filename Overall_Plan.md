**整个 Project 分解为多阶段行动指南**
根据项目文档（Project 3: Video Object Removal & Inpainting）的核心要求、实施路线图（Part 1~3）、提交规范以及当前计算机视觉行业研究（2023–2025 年 SOTA 论文如 ProPainter、SAM 系列、VGGT4D 等，已有成熟开源实现），我将整个项目拆分为 **7 个清晰、可执行的多阶段行动指南**。

每个阶段包含：
- **目标与对应项目部分**
- **所需工具/库/代码仓库**（基于官方最新实现，2026 年 3 月可用）
- **详细操作步骤**
- **成功标准**（定量 + 定性 + 项目预期）

**总体建议**：严格按照 CVPR LaTeX 模板撰写报告（6–8 页），所有相关论文必须全部引用（参考文献 [1]–[17]）。GitHub 仓库必须公开且 README 完整。Demo 视频必须打包成 `videos.zip` 提交 Canvas。建议每阶段结束后保存中间结果（mask、可视化、视频片段），便于报告和调试。灵活性允许自行优化，但 Part 1–3 必须全部完成，否则成绩受影响。

### **阶段 1: 准备阶段 — 文献综述与环境搭建**（1–2 周）
**目标**：理解核心概念（Optical Flow、Segmentation、Vision Transformer、Generative Inpainting），搭建统一开发环境。

**工具**：
- 阅读论文：项目参考文献列表 [1]–[17]（SAM 3 [1]、VGGT4D [6]、ProPainter [17] 等）。
- 开发环境：Python 3.10+、PyTorch 2.0+、CUDA 12.x（推荐）。
- 核心库：OpenCV、NumPy、Matplotlib、Ultralytics（YOLO）。

**步骤**：
1. 下载所有推荐论文 PDF 并阅读（重点：Mask R-CNN、SAM 系列、ProPainter）。
2. 创建 conda 环境：`conda create -n video-inpaint python=3.10` 并安装 PyTorch。
3. 克隆必要仓库（后续阶段会用到）：
   - ProPainter：`git clone https://github.com/sczhou/ProPainter.git`
   - SAM 2：`git clone https://github.com/facebookresearch/sam2.git`
   - Track-Anything：`git clone https://github.com/gaomingqi/Track-Anything.git`
   - VGGT4D：`git clone https://github.com/3DAgentWorld/VGGT4D.git`
   - SAM 3（2025 新版）：`git clone https://github.com/facebookresearch/sam3.git`
   - YOLOv8：`pip install ultralytics`

**成功标准**：
- 完成所有参考文献阅读笔记（至少 1 页总结）。
- 环境可运行 ProPainter 官方 demo（无报错）。
- 理解 Part 1–3 技术路线，能用自己的话解释“动态掩码提取 + 时序 Inpainting”。

### **阶段 2: 数据集准备**（0.5 周）
**目标**：获取并预处理所有必需数据。

**工具**：
- 必选数据集：
  - Wild Video：自行拍摄（走廊行人）或用 SORA 类模型生成。
  - Sample Data：课程提供的 bmx-trees 和 tennis 场景。
  - DAVIS 2017（强烈推荐用于高分）：官方下载 https://davischallenge.org/ （480p TrainVal + GT masks）。
- 工具：FFmpeg（视频切帧/合并）、Roboflow 或 Labelme（可选标注验证）。

**步骤**：
1. 拍摄/下载 Wild Video（至少 10–30 秒，含动态物体）。
2. 下载 bmx-trees、tennis 和 DAVIS 数据集。
3. 统一格式：所有视频转成 MP4，帧率为 30fps，分辨率一致（推荐 480p/720p）。

**成功标准**：
- 拥有 3 个必选视频 + 至少 1 个 DAVIS 序列。
- 所有视频能正常读取（OpenCV VideoCapture 测试通过）。
- 准备好 Ground Truth（DAVIS）用于后续 PSNR/SSIM 计算。

### **阶段 3: Part 1 实现 — 传统手工艺 Baseline**（1 周）
**目标**：用经典 CV 方法实现完整 pipeline，理解底层逻辑。

**工具**：
- 掩码提取：Ultralytics YOLOv8-Seg（`yolov8s-seg.pt`）或 Detectron2 Mask R-CNN。
- 动态判断：OpenCV Sparse Optical Flow（Lucas-Kanade）。
- Inpainting：`cv2.inpaint`（Telea/Navier-Stokes） + 自定义时序传播（前/后帧借用像素）。
- 辅助：OpenCV dilation（膨胀掩码）。

**步骤**：
1. 用 YOLOv8-Seg 提取 Person/Bicycle 等动态类掩码。
2. 对掩码内特征点计算光流运动幅度，阈值过滤静态物体。
3. 膨胀掩码 + 时序传播（优先同位置前/后帧干净像素）+ cv2.inpaint 作为 fallback。
4. 输出处理后视频。

**成功标准**：
- 在简单静态背景视频上有效（视觉上背景干净）。
- 复杂纹理区域可能出现模糊（符合项目“Expected Result”）。
- 产生中间可视化（mask 叠加、光流图、修复前后对比）用于报告 flowchart。

### **阶段 4: Part 2 实现 — SOTA AI 驱动管道**（1.5–2 周）
**目标**：复现最新开源方法，实现高质量动态物体移除。

**工具**（任选其一掩码 + ProPainter Inpainting）：
- **动态掩码提取**（必须选 1 个）：
  - Option A：Track-Anything（https://github.com/gaomingqi/Track-Anything）—— SAM + XMem 交互式跟踪。
  - Option B：SAM 2（https://github.com/facebookresearch/sam2）—— 实时视频分割。
  - Option C：VGGT4D（https://github.com/3DAgentWorld/VGGT4D）—— 零样本 Vision Transformer 动态线索挖掘（Gram Similarity）。
- **Inpainting**：ProPainter（https://github.com/sczhou/ProPainter）—— Dual-domain Propagation + Sparse Transformer（ICCV 2023 SOTA）。
  - 备选：E2FGVI 或 FGVC（项目允许）。

**步骤**：
1. 安装对应仓库依赖（严格按 README）。
2. 用选定掩码方法生成高质量动态 mask（支持视频输入）。
3. 将 mask 输入 ProPainter，生成修复视频。
4. 保存每帧 mask 和修复结果。

**成功标准**：
- 处理大遮挡时纹理清晰，明显优于 Part 1（项目预期）。
- 在 DAVIS 上 IoU mean/recall 有提升。
- 生成高质量视频（无明显 artifact），用于 Demo。

### **阶段 5: Part 3 实现 — 优化与扩展**（1–1.5 周，必须完成至少 1 个方向）
**目标**：发现局限并改进，体现探索性。

**工具**（任选 1 个方向）：
- Direction A：SAM 3（https://github.com/facebookresearch/sam3）替换 Track-Anything 骨干，或用 VGGT4D 输出作为 SAM 3 prompt 精炼掩码。
- Direction B：将 VGGT4D 移植到更强 3D 基础模型（如 Pi3 或 MapAnything）。
- Direction C：Diffusion 模型（Stable Diffusion Inpainting + ControlNet）修复关键帧，再传播（项目推荐 [12][16]）。

**步骤**：
1. 分析 Part 2 失败案例（artifact、不一致掩码）。
2. 实现至少 1 个改进方向，生成新版本 pipeline。
3. 对比前后结果（A/B 测试）。

**成功标准**：
- 至少 1 个方向有明显提升（IoU ↑ 或 视觉 artifact ↓）。
- 报告中必须展示 failure cases + 改进理由（否则成绩扣分）。
- 产生对比图表（用于 Experiments 部分）。

### **阶段 6: 实验评估与可视化**（1 周）
**目标**：量化 + 定性分析，准备报告素材。

**工具**：
- 指标：IoU mean (JM) / IoU recall (JR)（参考 VGGT4D 论文）、PSNR & SSIM（仅 DAVIS，参考 ProPainter）。
- 库：scikit-image、OpenCV 计算指标；Matplotlib/Seaborn 绘图。
- 可视化：高品质 flowchart（draw.io 或 TikZ）、修复前后对比帧、failure cases。

**步骤**：
1. 在所有必选数据集 + DAVIS 上跑 3 个 Part 的 pipeline。
2. 计算表格指标，生成定性对比图。
3. 额外数据实验（推荐）以冲击高分。

**成功标准**：
- 表格完整（至少 3 个方法 × 多个数据集）。
- 视觉效果“clean and aesthetic”（项目强调）。
- 所有必选视频处理完成。

### **阶段 7: 报告撰写、代码与演示提交**（1 周）
**目标**：完成所有提交要求。

**工具**：
- 报告：CVPR LaTeX 模板（arXiv 上传可选，记得改成 CAMERA-READY）。
- 代码：公共 GitHub（README 包含依赖、运行命令、权重、视觉结果）。
- Demo：处理后视频 → `videos.zip`（Wild + bmx-trees + tennis）。

**步骤**：
1. 报告结构严格按要求：Abstract（含 GitHub 链接）→ Related Work（必须全引）→ Method（flowchart）→ Experiments（表格+图）→ Conclusion。
2. 上传 arXiv（可选但推荐）。
3. GitHub README 完整 + 视频结果。
4. Canvas 提交 PDF + videos.zip。

**成功标准**：
- 报告 6–8 页，引用全部 [1]–[17]，GitHub 可运行。
- Demo 视频清晰、无水印。
- 完整性：Part 1–3 全部完成 + 额外实验 = 高分潜力。

**时间总计建议**：6–8 周（小组协作：一人负责掩码，一人负责 Inpainting + 报告）。

**当前行业研究补充建议**（2026 年视角）：
- SAM 3 与 VGGT4D 代表 2025 年零样本/概念驱动分割最新进展，已开源。
- ProPainter 仍是视频 Inpainting 经典基准，但 Diffusion + ControlNet（Direction C）是当前生成式修复主流，可进一步结合最新视频 Diffusion 模型（如 Sora 风格）扩展。
- 推荐额外实验：用 DAVIS 以外的 YouTube-8M 或自定义复杂场景验证泛化性。

严格按此指南执行，可确保覆盖项目所有必选要求，同时体现“当前行业研究”深度。遇到具体代码问题，可在 GitHub issue 或官方 demo 中查阅最新解决方案。祝项目顺利，拿到高分！