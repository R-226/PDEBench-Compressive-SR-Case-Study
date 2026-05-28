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

def stack_time_as_channels(x):
    """把时序维度堆叠为通道维度.
    输入: [B, T, H, W, V] → 输出: [B, T*V, H, W]"""
    assert x.dim() == 5, f"Expected 5D input, got {x.dim()}D"
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

        # === 4. 时间维度扩展 ===
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

        # 方法2: 时序MLP建模非线性演化
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

def full_loss(pred, target, known_t=[1,5], lambda_pde=10.0):
    B, T, H, W, C = pred.shape

    # 1. 已知时刻监督（强监督）
    mse_known = 0.0
    for t in known_t:
        mse_known += F.mse_loss(pred[:, t], target[:, t])
    mse_known /= len(known_t)

    # 2. 全序列物理约束（弱监督但全覆盖）
    pde_res = compute_full_pde_residual(pred)  # 需要实现对整个10步的残差

    return mse_known + lambda_pde * pde_res

def compute_full_pde_residual(pred, dx=1.0/128.0, dt=1.0, Du=1e-3, Dv=5e-3, alpha=5e-3, beta=1):
    """
    计算反应-扩散方程在整个时间序列上的物理残差。

    方程：
      ∂u/∂t = Du ∇²u + u - u³ - v + α
      ∂v/∂t = Dv ∇²v + β(u - v)

    参数:
      pred: [B, T, H, W, C=2]，预测的 u, v 场
      dx, dt: 空间和时间步长（根据 PDEBench 设置调整）
      Du, Dv, alpha, beta: 反应-扩散参数（需与数据生成一致）

    返回:
      标量：平均 PDE 残差（L2）
    """
    B, T, H, W, C = pred.shape
    assert C == 2, "Expected 2 channels (u, v)"
    assert T >= 3, "Need at least 3 time steps for central difference"

    u = pred[..., 0]  # [B, T, H, W]
    v = pred[..., 1]  # [B, T, H, W]

    # === 时间导数 ∂/∂t：使用中心差分（内部点），边界用前向/后向差分 ===
    ut = torch.zeros_like(u)
    vt = torch.zeros_like(v)

    # 内部点 (t=1 to T-2)
    ut[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dt)
    vt[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dt)

    # 边界：t=0（前向差分），t=T-1（后向差分）
    ut[:, 0] = (u[:, 1] - u[:, 0]) / dt
    ut[:, -1] = (u[:, -1] - u[:, -2]) / dt
    vt[:, 0] = (v[:, 1] - v[:, 0]) / dt
    vt[:, -1] = (v[:, -1] - v[:, -2]) / dt

    # === 拉普拉斯 ∇²：5-point stencil，带边界复制 ===
    def laplacian_2d(field):
        # field: [B, T, H, W]
        padded = F.pad(field, (1, 1, 1, 1), mode='replicate')  # [B, T, H+2, W+2]
        center = padded[:, :, 1:-1, 1:-1]
        left = padded[:, :, 1:-1, :-2]
        right = padded[:, :, 1:-1, 2:]
        top = padded[:, :, :-2, 1:-1]
        bottom = padded[:, :, 2:, 1:-1]
        return (left + right + top + bottom - 4 * center) / (dx ** 2)

    lap_u = laplacian_2d(u)  # [B, T, H, W]
    lap_v = laplacian_2d(v)

    # === 反应项 ===
    reaction_u = u - u**3 - v - alpha
    reaction_v = beta * (u - v)

    # === 残差 ===
    res_u = ut - (Du * lap_u + reaction_u)
    res_v = vt - (Dv * lap_v + reaction_v)

    # 返回平均 L2 残差
    loss = torch.mean(res_u**2 + res_v**2)
    return loss

if __name__ == "__main__":
    # ------------------------
    # 配置参数
    # ------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5_path = "../data/2D_diff-react_NA_NA.h5"
    batch_size = 4
    num_epochs = 200
    lr = 1e-3
    log_dir = "./runs/upsampler_pretrain"
    save_path = "./ckpts/upsampler.pth"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = SummaryWriter(log_dir)

    # ------------------------
    # 数据加载
    # ------------------------
    print("Loading dataset...")
    dataset = PDEUpsampleDataset(h5_path=h5_path, step=5, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ------------------------
    # 模型 & 优化器
    # ------------------------
    model = PDEFullUpsampler(in_t=2, out_t=10, in_s=32, out_s=128, c=2, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ------------------------
    # 训练循环
    # ------------------------
    print(f"Start training on {device}...")
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for low_res, high_res in dataloader:
            # low_res: [B, 2, 32, 32, 2]
            # high_res: [B, 10, 128, 128, 2]

            optimizer.zero_grad()
            pred = model(low_res)  # [B, 10, 128, 128, 2]

            # 使用已知时刻（t=0 和 t=1，对应原 t=0 和 t=5）做监督
            # high_res 是 t=0~9，low_res 是下采样后 t=0,1 → 对应 high_res 的 t=0,5
            # 所以 known_t 应为 [0, 5]
            loss = full_loss(pred, high_res, known_t=[0, 5], lambda_pde=5.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        # 每 20 轮保存一次
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), save_path.replace(".pth", f"_epoch{epoch+1}.pth"))

    # 最终保存
    torch.save(model.state_dict(), save_path)
    print(f"Training finished. Model saved to {save_path}")
    writer.close()