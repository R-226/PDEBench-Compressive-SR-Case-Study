import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import torch
import torch.nn.functional as F
import h5py
import numpy as np
import pickle
import os
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn as nn
from pdebench.models.unet.unet import UNet2d
from pdebench.models.fno.fno import FNO2d
import argparse
from torch.utils.tensorboard import SummaryWriter
from pdebench.models.unet.utils import UNetDatasetMult
from pdebench.models.fno.utils import FNODatasetMult
from pdebench.models.metrics import metric_func
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)


initial_step = 10
mode = "Unet"
model_name = "2D_diff-react_upsample_expert"
Lx=1.0
Ly=1.0
Lz=1.0


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
        # x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, 2, 2, 32, 32]
        """
        x: [B, H=32, W=32, T_in=2, C=2]
        return: [B, H=128, W=128,T_out=10, C=2]
        """
        # 转为 [B, C, T, H, W]
        x = x.permute(0, 4, 3, 1, 2).contiguous()

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
        # out = (out + res).permute(0, 2, 3, 4, 1).contiguous()  # [B, 10, 128, 128, 2]
        out = (out + res).permute(0, 3, 4, 2, 1).contiguous() # [B, 128, 128, 10, 2]
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

class ScaleAlignLoss(nn.Module):
    """尺度对齐Loss：强迫重建值的均值/方差和真实值一致，拉回真实范围"""
    def forward(self, pred, target):
        # 1. 计算真实值和重建值的均值、标准差
        target_mean = target.mean()
        target_std = target.std() + 1e-6  # 防除0
        pred_mean = pred.mean()
        pred_std = pred.std() + 1e-6

        # 2. 尺度校准：让重建值的均值=真实值均值，标准差=真实值标准差
        pred_scaled = (pred - pred_mean) * (target_std / pred_std) + target_mean

        # 3. 计算校准后的L1损失（对尺度更敏感，不会压平波动）
        scale_loss = F.l1_loss(pred_scaled, target)
        return scale_loss

class ScaleAlignLoss(nn.Module):
    """尺度对齐Loss：强迫重建值的均值/方差和真实值一致，拉回真实范围"""
    def forward(self, pred, target):
        # 1. 计算真实值和重建值的均值、标准差
        target_mean = target.mean()
        target_std = target.std() + 1e-6  # 防除0
        pred_mean = pred.mean()
        pred_std = pred.std() + 1e-6

        # 2. 尺度校准：让重建值的均值=真实值均值，标准差=真实值标准差
        pred_scaled = (pred - pred_mean) * (target_std / pred_std) + target_mean

        # 3. 计算校准后的L1损失（对尺度更敏感，不会压平波动）
        scale_loss = F.l1_loss(pred_scaled, target)
        return scale_loss

# 配合普通L1Loss，替换原来的MSE
class CombinedPretrainLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.scale_align = ScaleAlignLoss()
        self.high_freq = self._high_frequency_loss  # 加高频细节Loss，还原小斑块

    def _high_frequency_loss(self, pred, target):
        """高频细节Loss：提取真实值和重建值的高频分量，强迫还原小斑块"""
        # 用sobel算子提取高频（边缘/小斑块）
        sobel_x = torch.tensor([[-1,0,1], [-2,0,2], [-1,0,1]], dtype=pred.dtype, device=pred.device)
        sobel_x = sobel_x.view(1,1,3,3).repeat(pred.size(1), 1, 1, 1)  # 适配通道数
        sobel_y = torch.tensor([[-1,-2,-1], [0,0,0], [1,2,1]], dtype=pred.dtype, device=pred.device)
        sobel_y = sobel_y.view(1,1,3,3).repeat(pred.size(1), 1, 1, 1)

        # 计算高频分量
        pred_hf_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.size(1))
        pred_hf_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.size(1))
        target_hf_x = F.conv2d(target, sobel_x, padding=1, groups=target.size(1))
        target_hf_y = F.conv2d(target, sobel_y, padding=1, groups=target.size(1))

        # 高频分量的L1损失
        hf_loss = (F.l1_loss(pred_hf_x, target_hf_x) + F.l1_loss(pred_hf_y, target_hf_y)) / 2
        return hf_loss

    def forward(self, pred, target):
        """组合Loss：L1（尺度） + 尺度对齐（范围） + 高频（细节）"""
        # 调整维度：[B, V, T, H, W] → [B*T, V, H, W]（适配2D卷积）
        pred_2d = pred.reshape(-1, pred.size(1), pred.size(3), pred.size(4))
        target_2d = target.reshape(-1, target.size(1), target.size(3), target.size(4))

        l1_loss = self.l1(pred, target)  # 基础L1，不压平波动
        scale_loss = self.scale_align(pred_2d, target_2d)  # 拉回真实尺度
        hf_loss = self.high_freq(pred_2d, target_2d)  # 还原小斑块

        # 权重：L1为主，尺度和高频辅助（可调）
        total_loss = 0.5 * l1_loss + 0.3 * scale_loss + 0.2 * hf_loss
        return total_loss

class ConservationLoss(nn.Module):
    def __init__(self, relative=True, eps=1e-6):
        super().__init__()
        self.relative = relative
        self.eps = eps

    def forward(self, pred, gt):
        """
        鼓励空间积分（总量）守恒。
        输入:
            pred, gt: [B, T, H, W] 或 [B, T, H, W, C]
        """
        # 自动判断是否有通道维
        if pred.dim() == 5:
            # [B, T, H, W, C] → sum over H, W → [B, T, C]
            pred_int = pred.sum(dim=(2, 3))
            gt_int = gt.sum(dim=(2, 3))
        elif pred.dim() == 4:
            # [B, T, H, W] → [B, T]
            pred_int = pred.sum(dim=(2, 3))
            gt_int = gt.sum(dim=(2, 3))
        else:
            raise ValueError(f"Expected 4D or 5D input, got {pred.dim()}D")

        if self.relative:
            # 相对误差平方（更稳定）
            diff = pred_int - gt_int
            rel_diff = diff / (gt_int.abs() + self.eps)
            loss = (rel_diff ** 2).mean()
        else:
            # 绝对 MSE
            loss = F.mse_loss(pred_int, gt_int)

        return loss

def gaussian_kernel2d_2(kernel_size=5, sigma=1.0, channels=2, device="cuda"):
    ax = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel.to(device)

def smooth_and_downsample_2(x):
    """
    输入 x: [H, W, T, 1] 或 [H, W, T]（自动处理）
    输出: [32, 32, T, 1]
    """
    if x.dim() == 3:
        x = x.unsqueeze(-1)  # [H, W, T] -> [H, W, T, 1]
    H, W, T, C = x.shape
    assert C == 1, "最后一维应为1"

    device = x.device

    # 1. 重排为 [T, 1, H, W] —— 把时间步当作 batch，通道=1
    x = x.permute(2, 3, 0, 1)  # [H, W, T, 1] -> [T, 1, H, W]

    # 2. 高斯平滑（每个时间步独立）
    kernel = gaussian_kernel2d_2(kernel_size=5, sigma=1.0, channels=1, device=device)  # [1, 1, 5, 5]
    x = F.conv2d(x, kernel, padding=2)  # [T, 1, H, W]

    # 3. 两次 2x 下采样 (128 -> 64 -> 32)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)  # [T, 1, 64, 64]
    x = F.avg_pool2d(x, kernel_size=2, stride=2)  # [T, 1, 32, 32]

    # 4. 恢复原始维度顺序: [32, 32, T, 1]
    x = x.permute(2, 3, 0, 1)  # [T, 1, 32, 32] -> [32, 32, T, 1]
    x = x.squeeze(-1)  # Remove the last dimension if it's 1
    return x

def time_downsample_fixed_step_mean_2(tensor, step=5):
    H, W, T = tensor.shape
    T_down = math.ceil(T / step)
    tensor_down = torch.zeros((H, W, T_down), dtype=tensor.dtype, device=tensor.device)
    for i in range(T_down):
        start = i * step
        end = min((i + 1) * step, T)
        tensor_down[ :, :, i] = torch.mean(tensor[:, :, start:end], dim=2)
    return tensor_down

def downsample(tensor):
    u = tensor[...,0]
    v = tensor[...,1]
    ud = []
    vd = []
    for i in u:
        i = smooth_and_downsample_2(i)
        i = time_downsample_fixed_step_mean_2(i)
        ud.append(i)
    for i in v:
        i = smooth_and_downsample_2(i)
        i = time_downsample_fixed_step_mean_2(i)
        vd.append(i)
    u = torch.stack(ud, dim=0)
    v = torch.stack(vd, dim=0)
    return torch.stack([u, v], dim=-1)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h5_path = "../data/2D_diff-react_NA_NA.h5"
model_upsample_path = "./ckpts/upsampler.pth"
model_upsample = PDEFullUpsampler(in_t=2, out_t=10, in_s=32, out_s=128, c=2, hidden_dim=64).to(device)
model_upsample.load_state_dict(torch.load(model_upsample_path))

if mode == "Unet":
    model_name+="_UNet"
    model = UNet2d(initial_step * 2,2).to(device)
    checkpoint = torch.load("2D_diff-react_NA_NA_UNet-AR.pt", map_location=device)
elif mode == "FNO":
    model_name+="_FNO"
    model = FNO2d(
            num_channels=2,
            width=20,
            modes1=12,
            modes2=12,
            initial_step=initial_step,
        ).to(device)
    checkpoint = torch.load("2D_diff-react_NA_NA_FNO.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

if mode == "Unet":
    val_data = UNetDatasetMult(
            filename="2D_diff-react_NA_NA",
            # reduced_resolution=reduced_resolution,
            # reduced_resolution_t=reduced_resolution_t,
            # reduced_batch=reduced_batch,
            if_test=True,
            saved_folder="../data/",
        )
elif mode == "FNO":
    val_data = FNODatasetMult(
    filename="2D_diff-react_NA_NA",
    if_test=True,
    saved_folder="../data/",
    )

val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

if mode == "Unet":
    with torch.no_grad():
        for itot, (xx, yy) in enumerate(val_loader):
            logger.info(f"Processing batch {itot},each batch size: {xx.shape[0]}")
            xx = xx.to(device)  # noqa: PLW2901
            yy = yy.to(device)  # noqa: PLW2901
            temp = yy[..., :initial_step, :]
            temp = downsample(temp)
            pred = model_upsample(temp)
            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)

            for _t in range(initial_step, yy.shape[-2]):
                inp = xx.reshape(inp_shape)
                temp_shape = [0, -1]
                temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                inp = inp.permute(temp_shape)

                temp_shape = [0]
                temp_shape.extend(list(range(2, len(inp.shape))))
                temp_shape.append(1)
                im = model(inp).permute(temp_shape).unsqueeze(-2)
                pred = torch.cat((pred, im), -2)
                xx = torch.cat((xx[..., 1:, :], im), dim=-2)  # noqa: PLW2901
            (
                _err_RMSE,
                _err_nRMSE,
                _err_CSV,
                _err_Max,
                _err_BD,
                _err_F,
            ) = metric_func(
                pred,
                yy,
                if_mean=True,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                initial_step=initial_step,
            )


            if itot == 0:
                err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = (
                    _err_RMSE,
                    _err_nRMSE,
                    _err_CSV,
                    _err_Max,
                    _err_BD,
                    _err_F,
                )
                pred_plot = pred[:1]
                target_plot = yy[:1]
                val_l2_time = torch.zeros(yy.shape[-2]).to(device)
            else:
                err_RMSE += _err_RMSE
                err_nRMSE += _err_nRMSE
                err_CSV += _err_CSV
                err_Max += _err_Max
                err_BD += _err_BD
                err_F += _err_F

                mean_dim = list(range(len(yy.shape) - 2))
                mean_dim.append(-1)
                mean_dim = tuple(mean_dim)
                val_l2_time += torch.sqrt(
                    torch.mean((pred - yy) ** 2, dim=mean_dim)
                )
elif mode == "FNO":
    with torch.no_grad():
        itot = 0
        for itot, (xx, yy, grid) in enumerate(val_loader):
            logger.info(f"Processing batch {itot},each batch size: {xx.shape[0]}")
            xx = xx.to(device)  # noqa: PLW2901
            yy = yy.to(device)  # noqa: PLW2901
            grid = grid.to(device)  # noqa: PLW2901

            temp = yy[..., :initial_step, :]
            temp = downsample(temp)
            pred = model_upsample(temp)
            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)

            for _t in range(initial_step, yy.shape[-2]):
                inp = xx.reshape(inp_shape)
                im = model(inp, grid)
                pred = torch.cat((pred, im), -2)
                xx = torch.cat((xx[..., 1:, :], im), dim=-2)  # noqa: PLW2901

            (
                _err_RMSE,
                _err_nRMSE,
                _err_CSV,
                _err_Max,
                _err_BD,
                _err_F,
            ) = metric_func(
                pred,
                yy,
                if_mean=True,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                initial_step=initial_step,
            )
            if itot == 0:
                err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = (
                    _err_RMSE,
                    _err_nRMSE,
                    _err_CSV,
                    _err_Max,
                    _err_BD,
                    _err_F,
                )
                pred_plot = pred[:1]
                target_plot = yy[:1]
                val_l2_time = torch.zeros(yy.shape[-2]).to(device)
            else:
                err_RMSE += _err_RMSE
                err_nRMSE += _err_nRMSE
                err_CSV += _err_CSV
                err_Max += _err_Max
                err_BD += _err_BD
                err_F += _err_F

                mean_dim = list(range(len(yy.shape) - 2))
                mean_dim.append(-1)
                mean_dim = tuple(mean_dim)
                val_l2_time += torch.sqrt(
                    torch.mean((pred - yy) ** 2, dim=mean_dim)
                )

elif mode == "PINN":
    raise NotImplementedError



err_RMSE = np.array(err_RMSE.data.cpu() / itot)
err_nRMSE = np.array(err_nRMSE.data.cpu() / itot)
err_CSV = np.array(err_CSV.data.cpu() / itot)
err_Max = np.array(err_Max.data.cpu() / itot)
err_BD = np.array(err_BD.data.cpu() / itot)
err_F = np.array(err_F.data.cpu() / itot)
logger.info(f"RMSE: {err_RMSE:.5f}")
logger.info(f"normalized RMSE: {err_nRMSE:.5f}")
logger.info(f"RMSE of conserved variables: {err_CSV:.5f}")
logger.info(f"Maximum value of rms error: {err_Max:.5f}")
logger.info(f"RMSE at boundaries: {err_BD:.5f}")
logger.info(f"RMSE in Fourier space: {err_F}")
val_l2_time = val_l2_time / itot

errs = (err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F)
if mode == "Unet":
    pickle_path = Path("2D_diff-react_NA_NA_Unet-down&up.pickle")
elif mode == "FNO":
    pickle_path = Path("2D_diff-react_NA_NA_FNO-down&up.pickle")
pickle.dump(errs, pickle_path.open("wb"))

channel_plot = 0
x_min = -1
x_max = 1
y_min = -1
y_max = 1
t_min = 0
t_max = 5
dim = len(yy.shape) - 3
plt.ioff()

fig, ax = plt.subplots(figsize=(6.5, 6))
h = ax.imshow(
    pred_plot[..., -1, channel_plot].squeeze().t().detach().cpu(),
    extent=[x_min, x_max, y_min, y_max],
    origin="lower",
    aspect="auto",
)
h.set_clim(
    target_plot[..., -1, channel_plot].min(),
    target_plot[..., -1, channel_plot].max(),
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=30)
ax.set_title("Prediction", fontsize=30)
ax.tick_params(axis="x", labelsize=30)
ax.tick_params(axis="y", labelsize=30)
ax.set_ylabel("$y$", fontsize=30)
ax.set_xlabel("$x$", fontsize=30)
plt.tight_layout()
filename = model_name + "_pred.pdf"
plt.savefig(filename)

fig, ax = plt.subplots(figsize=(6.5, 6))
h = ax.imshow(
    target_plot[..., -1, channel_plot].squeeze().t().detach().cpu(),
    extent=[x_min, x_max, y_min, y_max],
    origin="lower",
    aspect="auto",
)
h.set_clim(
    target_plot[..., -1, channel_plot].min(),
    target_plot[..., -1, channel_plot].max(),
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=30)
ax.set_title("Data", fontsize=30)
ax.tick_params(axis="x", labelsize=30)
ax.tick_params(axis="y", labelsize=30)
ax.set_ylabel("$y$", fontsize=30)
ax.set_xlabel("$x$", fontsize=30)
plt.tight_layout()
filename = model_name + "_data.pdf"
plt.savefig(filename)