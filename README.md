# HW_hackerthon_release

华为黑客松 3DGS 重建提交代码。统一流程：官方预处理与均匀 16 视角选择 → 冻结 ReSplat → B 中修正 SH1 导出 → 稀疏 Adam 优化 5000 步 → LPIPS 尾段优化 300 步。

尾段使用全图损失 `0.8 L1 + 0.2 (1−SSIM) + 0.1 VGG-LPIPS`。各参数组学习率乘 0.3 后固定，延续优化器动量和视图采样状态，保持高斯拓扑不变。优化超参数集中在 [config.json](config.json)，跨场景统一。

## 安装与运行

运行环境：Linux、Python 3.12、NVIDIA GPU、CUDA 12.8 开发工具链、Git 和 C++ 编译器。缺少相机位姿的场景还需安装 COLMAP。在独立虚拟环境中执行：

```bash
python setup.py
python run_pipeline.py --scene /path/to/scene --output /path/to/new-output
```

`setup.py` 安装依赖，并将固定版本的外部源码、ReSplat 权重和 LPIPS 所需的 VGG 权重下载到对应资源目录。外部源码与 ReSplat 权重默认位于 `assets/`；VGG 权重使用 PyTorch 缓存。

已有资源或外部仓库不可访问时，可通过环境变量指定固定版本的完整 checkout 和权重：

| 环境变量 | 资源 | 使用阶段 |
|---|---|---|
| `RESPLAT_VENDOR` | 官方源码目录 | 安装与运行 |
| `RESPLAT_CHECKPOINT` | ReSplat 权重文件 | 安装与运行 |
| `FASTGS_SOURCE` | FastGS 源码目录 | 安装 |
| `FASTERGS_SOURCE` | FasterGS 源码目录 | 安装 |

固定版本和权重 SHA256 见 [resources.json](resources.json)。程序在重建前后校验 ReSplat 源码与权重。

输入为官方预处理支持的场景目录，可包含 COLMAP 位姿、`transforms.json` 或原始图片。测试划分固定为 `llffhold=8`，测试 RGB 仅用于重建后的评估。`--gpu` 指定物理卡号或 UUID；省略时遵循 `CUDA_VISIBLE_DEVICES`。

输出路径必须是新目录。主要产物为 `scene.ply`（标准 3DGS SH3）、`rgb/*.png`、`metrics.json`、`report.json` 和 `report.md`。`work/`、`resplat/` 及阶段日志记录中间结果。

## 时间与评分

报告分别记录 A、模型加载、前向、B、渲染和质量评估耗时。B 包括外部 SH 修正、优化、PLY 导出，以及冻结前向包装器中除模型加载和推理外的开销。完整流程计时截至 PLY 和 RGB PNG 交付，质量评估单独计时。

重建完成后，在同一 GPU 空闲时运行标定，补齐原报告的时间分和总分：

```bash
python calibrate.py --work /path/to/previous-output/work --output /path/to/calibration --gpu 0 --report /path/to/previous-output/report.json
```

已有匹配标定时，在上述命令中添加 `--reuse /path/to/existing/calibration.json`。新运行也可直接指定参考：

```bash
python run_pipeline.py --scene /path/to/scene --output /path/to/new-output --gpu 0 --reference /path/to/calibration/calibration.json
```

参考采用官方 FastGS 随机初始化训练 30000 步、`mult=0.5`，`T_iter=训练循环时间/30000`。程序核对 GPU、相机、划分和参考协议，要求 `B < 30000 × T_iter`；达到或超过预算时，入口返回 2，报告状态为 `budget_exceeded`。缺少匹配参考时，时间分和总分留空。

评分暂按场景均值及区间截断计算，正式聚合规则以赛方最终细则为准。共享负载下的重建耗时与空闲标定应分开分析；补测标定只更新参考时间，不会重新测量原运行的 B 耗时。

## 外部依赖与提交材料

ReSplat 与官方 3DGS 代码保持冻结。FastGS 提供绝对梯度渲染，FasterGS 提供稀疏 Adam，gsplat 用于前向及早期几何修复，LPIPS 使用冻结 VGG。依赖来源和版本见 [resources.json](resources.json)，许可证保留在各外部仓库中。

提交材料还包括指导书第 5、7 页要求的方案介绍、演示、自验证结果和 AI 使用记录。运行入口生成自验证报告；代码整理使用 Codex 辅助，具体 AI 模型和 Token 消耗按团队的平台记录汇总。

[validation.json](validation.json) 保存 NVIDIA H20 开发环境中 DL3DV 001dccbc、billboard、counter 的单轮自验证及对应 FastGS 标定。干净机器上的依赖编译验证待完成。
