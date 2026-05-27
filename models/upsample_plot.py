import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import torch
import torch.nn.functional as F
import h5py
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn as nn
from pdebench.models.unet.unet import UNet2d
import argparse
from torch.utils.tensorboard import SummaryWriter

def visualize_upsample_result(
    model,
    low_res_input,      # [2, 32, 32, 2] —— 单个样本，无 batch 维度
    high_res_target,    # [10, 128, 128, 2] —— 单个样本，无 batch 维度
    time_idxs=[0, 5],   # 要可视化的时刻（必须是 0~9）
    channel=0,          # 0: u 场, 1: v 场
    save_dir='./figs'
):
    """
    可视化单个样本的上采样效果。
    """
    import torch
    import os

    model.eval()
    device = next(model.parameters()).device

    # 添加 batch 维度 → [1, 2, 32, 32, 2]
    low_res_batch = low_res_input.unsqueeze(0).to(device)
    high_res_batch = high_res_target.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_batch = model(low_res_batch)  # [1, 10, 128, 128, 2]

    # 移除 batch 维度
    pred = pred_batch.squeeze(0).cpu()        # [10, 128, 128, 2]
    gt_full = high_res_target.cpu()           # [10, 128, 128, 2]
    low_res = low_res_input.cpu()             # [2, 32, 32, 2]

    os.makedirs(save_dir, exist_ok=True)

    for t in time_idxs:
        if t < 0 or t >= 10:
            print(f"Warning: time_idx {t} out of range [0,9], skipped.")
            continue

        # 真实值
        gt = gt_full[t, :, :, channel].numpy()      # [128, 128]

        # 模型重建
        recon = pred[t, :, :, channel].numpy()      # [128, 128]
        low_upsampled = None
        # 找到对应的低分辨率帧（t=0→low[0], t=5→low[1]）
        if t == 0:
            low_frame = low_res[0, :, :, channel]   # [32, 32]
        elif t == 5:
            low_frame = low_res[1, :, :, channel]   # [32, 32]
        else:
            # 对于其他 t（如 t=2,7），我们无法从 low_res 直接获取，
            # 所以跳过“降采样输入”图，或用插值近似（这里简单跳过对比图）
            low_upsampled = None

        if low_upsampled is not None or t in [0, 5]:
            if t in [0, 5]:
                # 上采样低分辨率输入用于对比（最近邻）
                low_upsampled = F.interpolate(
                    low_frame.unsqueeze(0).unsqueeze(0),  # [1,1,32,32]
                    size=(128, 128),
                    mode='nearest'
                ).squeeze().numpy()  # [128, 128]

            # 绘图
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            titles = ["Ground Truth", "Downsampled Input (↑)", "Model Reconstruction"]
            images = [gt, low_upsampled, recon]
            cmap = 'viridis'

            for ax, img, title in zip(axes, images, titles):
                im = ax.imshow(img, cmap=cmap, origin='lower')
                ax.set_title(title, fontsize=12)
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            save_path = os.path.join(save_dir, f'sample_t{t}_{"u" if channel==0 else "v"}.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
        else:
            # 如果不是 t=0 或 t=5，只画 GT 和 Recon（因为没有对应的 low-res 输入）
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            titles = ["Ground Truth", "Model Reconstruction"]
            images = [gt, recon]
            cmap = 'viridis'

            for ax, img, title in zip(axes, images, titles):
                im = ax.imshow(img, cmap=cmap, origin='lower')
                ax.set_title(title, fontsize=12)
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            save_path = os.path.join(save_dir, f'sample_t{t}_{"u" if channel==0 else "v"}_no_lowres.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved (no low-res): {save_path}")
            plt.close()

def stack_time_as_channels(x):
    """
    适配你的数据维度：把时序维度堆叠为通道维度，供基准模型输入
    输入：x → [B, T, H, W, V]（批次、时间步、高、宽、通道）
          比如你的上采样输出：[B, 2, 128, 128, 2]
    输出：[B, T*V, H, W]（时间步×通道 作为新通道维度）
          比如输出：[B, 2×2=4, 128, 128]
    """
    # 确保输入是5维：[B, T, H, W, V]
    assert x.dim() == 5, f"输入维度必须是5维，当前是{x.dim()}维！"
    B, T, H, W, V = x.shape
    # 把时序T和通道V堆叠成新的通道维度：T*V
    x = x.permute(0, 1, 4, 2, 3)  # [B, T, V, H, W]
    x = x.reshape(B, T*V, H, W)    # [B, T*V, H, W]
    return x

def gaussian_kernel2d(kernel_size=5, sigma=1.0, channels=2, device="cuda"):
    ax = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel.to(device)

def smooth_and_downsample(x, out_size=32):
    T, H, W = x.shape
    device = x.device
    kernel = gaussian_kernel2d(kernel_size=5, sigma=1.0, channels=T, device=device)
    x = F.conv2d(x.unsqueeze(0), kernel, padding=2, groups=T).squeeze(0)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    return x

def time_downsample_fixed_step_mean(tensor, step=5):
    T, H, W = tensor.shape
    T_down = math.ceil(T / step)
    tensor_down = torch.zeros((T_down, H, W), dtype=tensor.dtype, device=tensor.device)
    for i in range(T_down):
        start = i * step
        end = min((i + 1) * step, T)
        tensor_down[i] = torch.mean(tensor[start:end], dim=0)
    return tensor_down

# 封装成数据集类（预训练专用）
class PDEUpsampleDataset(Dataset):
    """
    预训练数据集：
    - 输入：下采样后的 [2,32,32,2]（时间步1/5，空间32×32）
    - 标签：原始高分辨率 [2,128,128,2]（时间步1/5，空间128×128）
    """
    def __init__(self, h5_path, step=5, device="cuda"):
        self.h5_path = h5_path
        self.step = step
        self.device = device
        self.data_pairs = self._load_data()

    def _load_data(self):
        data_pairs = []
        with h5py.File(self.h5_path, "r") as h5_file:
            data_list = sorted(h5_file.keys())
            for seed_key in data_list:  # 遍历所有seed
                seed_group = h5_file[seed_key]
                data = np.array(seed_group["data"], dtype=np.float32)  # [T,128,128,2]
                data = torch.tensor(data, device=self.device)

                # 1. 提取到第10步
                t_indices = slice(0, 10)
                data_high = data[t_indices]  # [0:10,128,128,2] → 高分辨率标签

                # 2. 对每个通道下采样（u和v分开处理）
                data_low_u = smooth_and_downsample(data_high[..., 0])  # 调整维度适配下采样
                data_low_v = smooth_and_downsample(data_high[..., 1])
                data_low_u = time_downsample_fixed_step_mean(data_low_u, step=self.step)
                data_low_v = time_downsample_fixed_step_mean(data_low_v, step=self.step)

                # 3. 拼接u/v，恢复维度 [2,32,32,2]
                data_low = torch.stack([data_low_u, data_low_v], dim=-1)

                # 4. 保存高低分辨率对
                data_pairs.append((data_low, data_high))
        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        return self.data_pairs[idx]

class PDEFullUpsampler(nn.Module):
    """输入 [B,2,32,32,2] → 输出 [B,10,128,128,2]"""
    def __init__(self, in_t=2, out_t=10, in_s=32, out_s=128, c=2, hidden_dim=64):
        super().__init__()
        self.in_t = in_t
        self.out_t = out_t
        self.c = c
        # === 1. 输入reshape: [B, T, H, W, C] → [B, C, T, H, W] ===
        # === 2. 编码器（时空特征提取）===
        self.enc1 = nn.Conv3d(c, hidden_dim, kernel_size=(3,3,3), padding=(1,1,1))
        self.enc2 = nn.Conv3d(hidden_dim, hidden_dim*2, kernel_size=(3,3,3), padding=(1,1,1))

        # === 3. 空间上采样（32→64→128）+ 时间扩展（2→10）===
        # 先空间上采样到128，再时间扩展（避免3D转置卷积显存爆炸）
        self.spatial_up1 = nn.ConvTranspose3d(
            hidden_dim*2, hidden_dim,
            kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)  # H,W: 32→64
        )
        self.spatial_up2 = nn.ConvTranspose3d(
            hidden_dim, hidden_dim//2,
            kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)  # 64→128
        )  # now [B, 32, 2, 128, 128]

        # === 4. 时间维度可学习扩展（核心！）===
        self.time_expander = LearnableTimeExpander(in_t=2, out_t=10, channels=hidden_dim//2)

        # === 5. 解码器（恢复通道）===
        self.dec = nn.Conv3d(hidden_dim//2, c, kernel_size=(3,3,3), padding=(1,1,1))

        # === 6. 残差路径（从输入直接上采样到输出尺寸）===
        self.residual = nn.Sequential(
            nn.Upsample(size=(out_t, out_s, out_s), mode='trilinear', align_corners=False),
            nn.Conv3d(c, c, kernel_size=1)
        )

    def forward(self, x):
        """
        x: [B, T_in=2, H=32, W=32, C=2]
        return: [B, T_out=10, H=128, W=128, C=2]
        """
        # 转为 [B, C, T, H, W]
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, 2, 2, 32, 32]

        # 主干
        feat = F.gelu(self.enc1(x))
        feat = F.gelu(self.enc2(feat))             # [B, 128, 2, 32, 32]
        feat = self.spatial_up1(feat)              # [B, 64, 2, 64, 64]
        feat = self.spatial_up2(feat)              # [B, 32, 2, 128, 128]
        feat = self.time_expander(feat)            # [B, 32, 10, 128, 128]
        out = self.dec(feat)                       # [B, 2, 10, 128, 128]

        # 残差
        res = self.residual(x)                     # [B, 2, 10, 128, 128]

        # 合并 & 转回原始格式
        out = (out + res).permute(0, 2, 3, 4, 1).contiguous()  # [B, 10, 128, 128, 2]
        return out

class LearnableTimeExpander(nn.Module):
    """将时间维度从 in_t 扩展到 out_t，通过可学习插值 + 动态建模"""
    def __init__(self, in_t=2, out_t=10, channels=32):
        super().__init__()
        self.in_t = in_t
        self.out_t = out_t

        # 方法1: 可学习插值核（基础）
        self.interp_weight = nn.Parameter(torch.randn(out_t, in_t))

        # 方法2: 加一个轻量时序MLP来建模非线性演化（关键！）
        self.temporal_mlp = nn.Sequential(
            nn.Conv3d(channels, channels*2, kernel_size=(1,1,1)),
            nn.GELU(),
            nn.Conv3d(channels*2, channels, kernel_size=(1,1,1))
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: [B, C, T_in, H, W]
        return: [B, C, T_out, H, W]
        """
        B, C, T_in, H, W = x.shape
        assert T_in == self.in_t

        # Step 1: 可学习插值
        w = self.softmax(self.interp_weight)  # [T_out, T_in]
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B, T_in, -1)  # [B, T_in, C*H*W]
        x_interp = torch.matmul(w, x_flat)    # [B, T_out, C*H*W]
        x_interp = x_interp.view(B, self.out_t, C, H, W).permute(0, 2, 1, 3, 4)  # [B, C, T_out, H, W]

        # Step 2: 用MLP建模物理演化（让中间帧不只是插值，而是“推演”）
        x_out = self.temporal_mlp(x_interp)
        return x_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h5_path = "../data/2D_diff-react_NA_NA.h5"
model_path = "./ckpts/upsampler.pth"
log_dir = "./figs"
writer = SummaryWriter(log_dir)
model = PDEFullUpsampler(in_t=2, out_t=10, in_s=32, out_s=128, c=2, hidden_dim=64).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
data = PDEUpsampleDataset(h5_path="../data/2D_diff-react_NA_NA.h5", step=5, device=device)
for idx, (data_low, data_high) in enumerate(data):
    print(f"样本 {idx}:")
    print(f"  低分辨率张量形状: {data_low.shape}")  # 预期 [2,32,32,2]
    print(f"  高分辨率张量形状: {data_high.shape}")  # 预期 [2,128,128,2]
    # 只打印前2个样本，避免输出过多
    if idx >= 1:
        break
    visualize_upsample_result(model=model, low_res_input=data_low, high_res_target=data_high)
    visualize_upsample_result(
    model=model,
    low_res_input=data_low,       # 不加 batch 维度！
    high_res_target=data_high,    # 不加 batch 维度！
    time_idxs=[0, 2, 5, 7, 9],    # 可视化多个时间步
    channel=0,                    # u 场
    save_dir="./figs/upsample_viz"
    )