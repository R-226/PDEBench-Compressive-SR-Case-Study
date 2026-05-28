实际上都是半年前弄的时候的东西，当时本来准备之后研究下原因再发，但是后来忙于别的事情一直没再管过，甚至代码等等东西也都丢了。不过前段时间翻到了之前的U盘发现里面还存着（3-2-1法则魅力时刻），感觉我之后也不一定有时间慢慢看了，还是直接发上来吧

---
# PDEBench Compressive Super-Resolution

对 2D diffusion-reaction 方程的高分辨率 PDE 数据进行极端压缩（160×）后，通过一个轻量超分辨率网络重建回原始分辨率，再送入专家模型求解。**实验表明，压缩→超分→求解这条链路的结果不仅没有精度损失，甚至略优于全分辨率直接求解。** FNO 推理 MSE 从 0.00951 降至 0.00913，Unet-AR 从 0.02516 降至 0.02415。

## 动机

将高分辨率 PDE 数据 `[10, 128, 128, 2]`（327,680 个浮点数）压缩为紧凑表示 `[2, 32, 32, 2]`（4,096 个浮点数，约 160× 压缩比），再训练一个超分辨率网络从紧凑表示重建原始分辨率，最后送入专家模型完成 PDE 求解。

```text
原始数据 [10, 128, 128, 2]    (327,680 values)
    │ 高斯平滑 + 4× AvgPool + 5× 时间合并
    ▼
压缩表示 [2, 32, 32, 2]      (4,096 values)
    │ PDEFullUpsampler (395K 参数)
    ▼
重建高分辨率 [10, 128, 128, 2]
    │ FNO / Unet-AR 专家模型
    ▼
PDE 求解结果
```

压缩策略：**高斯平滑**（σ=1.0, kernel=5）→ **4× 空间 AvgPool**（128→32）→ **5× 时间平均合并**（10→2）。工具代码见 [`pdebench/models/downsample_utils.py`](pdebench/models/downsample_utils.py)。

## 上采样模型

### 架构

```
PDEFullUpsampler: [B, 2, 32, 32, 2] → [B, 10, 128, 128, 2]

Input (Low-Res)
    │
    ▼
Conv3d Encoder (c→64→128)     ─┐
    │                           │
    ▼                           │
SpatialUp1 (32→64)             │
    │                           │
    ▼                           │
SpatialUp2 (64→128)            │
    │                           │
    ▼                           │
LearnableTimeExpander (2→10)   │
    │                           │
    ▼                           │
Conv3d Decoder (32→2)          │
    │                           │
    ▼                           │
    + ─── Residual (Trilinear Upsample + 1×1 Conv)
    │
    ▼
Output (High-Res)
```

**LearnableTimeExpander**：将时间维度从 2 帧扩展到 10 帧——

1. **可学习插值核**：softmax 归一化的权重矩阵 `[10, 2]`，对两帧低分辨率输入做加权插值
2. **时序 MLP**：两层 Conv3d（1×1×1 卷积 + GELU），对插值结果建模非线性物理演化

参数量约 **395K**，可在单张消费级 GPU 上完成训练。

### 损失函数设计

核心挑战：PDE 场数值跨度大且均值接近零，MSE 会驱使模型输出趋向均值。为此尝试了多种损失组合：

| 文件 | 策略 |
|------|------|
| `upsample_L2.py` | MSE + PDE 物理残差约束 |
| `upsample_L1_v1.py` | L1 + Soft ScaleAlign（校准均值/标准差后算 L1）+ Sobel 高频 |
| `upsample_L1_v2.py` | L1 + Hard ScaleAlign（惩罚均值/标准差偏离）|
| `upsample_L1_v3.py` | L1 + **RelativeScaleAlign**（对齐比例）+ Sobel 高频 |

各版本质量我也记不清了，好久之前的了，RelativeScaleAlign 对均值≈0 的数据最为稳定。所有版本均包含三类损失信号：

- **尺度对齐**：保持重建场的统计分布与真实场一致
- **高频细节**：Sobel 算子提取边缘/纹理，L1 约束高频一致性
- **物理约束**：反应-扩散方程残差（有限差分），作为弱监督信号

## 实验与结果

### 核心结果：压缩重建无损甚至更优

**关键发现**：将数据压缩 160× 经上采样重建后再送入专家模型，计算精度不低于全分辨率直接推理，甚至略有提升。

2D diffusion-reaction 任务，同一测试集：

| 模型 | 推理方式 | MSE ↓ | normMSE ↓ | Max Error ↓ |
|------|---------|-------|-----------|-------------|
| FNO | 全分辨率直接推理 | 0.00951 | 0.1405 | 0.1390 |
| FNO | **压缩→上采样→推理** | **0.00913** | **0.1349** | **0.1019** |
| Unet-AR | 全分辨率直接推理 | 0.02516 | 0.4270 | 0.1507 |
| Unet-AR | **压缩→上采样→推理** | **0.02415** | **0.4097** | **0.1215** |

### 失败过的路径

达到上述结果之前，尝试过两条更直观的思路，均未成功（记录在 `record/` 中）：

**直接训练**（`normal` → `nomal_final`, `grad_(noFFT)`, `grad and FFT`）

在降采样数据上直接训练 PDE 求解模型。多次调参后最优的 `nomal_final` 达到 MSE=0.088。引入梯度惩罚（`grad_(noFFT)`）试图抑制平坦化解，再叠加 FFT 惩罚（`grad and FFT`），MSE 仅小幅升至 0.093。问题不在数值——热力图揭露这些模型的输出几乎是均匀的平坦场，MSE 偏好使模型学会了输出"安全的平均值"。

**知识蒸馏**（`distill1` → `distil6(final)`）

以全分辨率基准模型为教师，降采样数据上的学生为蒸馏目标。反复调整蒸馏层、教师数据量、训练参数后，最优结果 `distill1` 仅达 MSE=0.099，`distil4` 增加教师数据反而恶化至 0.136。普遍出现花纹不一致、棋盘格伪影、平坦化等问题。

| 实验 | MSE ↓ | normMSE ↓ | 备注 |
|------|-------|-----------|------|
| normal_final | 0.088 | 1.217 | 直接训练最优（仍平坦化）|
| grad and FFT | 0.093 | 1.422 | 梯度+FFT 惩罚 |
| distill1 | 0.099 | 1.431 | 蒸馏最优 |
| distil6(final) | 0.104 | 1.393 | 蒸馏最终调参 |
| distil4 | 0.136 | 1.966 | 增加教师数据，反而更差 |

## 项目结构

```
pdebench/
├── models/
│   ├── downsample_utils.py      # 下采样工具（独立可复用）
│   ├── upsample_L2.py           # MSE + 物理约束
│   ├── upsample_L1_v1.py        # L1 + Soft ScaleAlign
│   ├── upsample_L1_v2.py        # L1 + Hard ScaleAlign
│   ├── upsample_L1_v3.py        # L1 + RelativeScaleAlign
│   ├── upsample_expert.py       # 上采样 + 专家模型完整评测
│   ├── upsample_plot.py         # 上采样结果可视化
│   ├── fno/  unet/  pinn/       # 上游 PDEBench 专家模型

record/
├── normal/  nomal_final/        # 直接训练（平坦解）
├── grad_(noFFT)/  grad and FFT/  # 梯度/FFT 惩罚（平坦解）
├── distill1 ~ distil6(final)/   # 知识蒸馏（失败）
└── upsample(pre)/ upsample(L1)/ # 上采样（均有良好结果）
    upsample_scale_epoch10/ upsample_scale_epoch50/
```

## 局限与未来方向

- **泛化性**：当前仅在 2D diffusion-reaction 上验证，需在其他 PDE 类型上确认现象普适性。
- **可解释性**：平滑下采样为何反而提升精度，目前仅有"隐式正则化"的假设，值得从频谱分析或信息瓶颈角度深入。
- **物理约束增强**：将守恒律嵌入上采样器作为硬约束而非软损失项，可能进一步提升质量。

## 许可证

本项目代码基于 MIT 许可证发布（[LICENSE.txt](LICENSE.txt)）。

PDEBench 数据集与上游代码版权归 NEC Labs Europe GmbH, Stuttgart University, CSIRO 及 PDEBench 贡献者所有（[LICENSE-PDEBench.txt](LICENSE-PDEBench.txt)）。

## 参考文献

- Takamoto, M., et al. *PDEBench: An Extensive Benchmark for Scientific Machine Learning*. 2024.
