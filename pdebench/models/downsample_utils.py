import torch
import torch.nn.functional as F
import h5py
import numpy as np
import math
from torch.utils.data import Dataset


def gaussian_kernel2d(kernel_size=5, sigma=1.0, channels=2, device="cuda"):
    ax = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel.to(device)


def smooth_and_downsample(x, out_size=32):
    """高斯平滑 + 两次 2x AvgPool，将 128x128 降采样到 32x32."""
    T, H, W = x.shape
    device = x.device
    kernel = gaussian_kernel2d(kernel_size=5, sigma=1.0, channels=T, device=device)
    x = F.conv2d(x.unsqueeze(0), kernel, padding=2, groups=T).squeeze(0)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    return x


def time_downsample_fixed_step_mean(tensor, step=5):
    """按固定步长做时间维度的均值合并，10 步 → 2 步."""
    T, H, W = tensor.shape
    T_down = math.ceil(T / step)
    tensor_down = torch.zeros((T_down, H, W), dtype=tensor.dtype, device=tensor.device)
    for i in range(T_down):
        start = i * step
        end = min((i + 1) * step, T)
        tensor_down[i] = torch.mean(tensor[start:end], dim=0)
    return tensor_down


class PDEUpsampleDataset(Dataset):
    """从 HDF5 加载高分辨率 PDE 数据，生成 (低分辨率输入, 高分辨率标签) 对.

    高分辨率: [10, 128, 128, 2] (10 个时间步, 2 通道)
    低分辨率: [2, 32, 32, 2]   (时间 5x 合并, 空间 4x 降采样)
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
            for seed_key in data_list:
                seed_group = h5_file[seed_key]
                data = np.array(seed_group["data"], dtype=np.float32)  # [T, 128, 128, 2]
                data = torch.tensor(data, device=self.device)

                t_indices = slice(0, 10)
                data_high = data[t_indices]  # [10, 128, 128, 2]

                data_low_u = smooth_and_downsample(data_high[..., 0])
                data_low_v = smooth_and_downsample(data_high[..., 1])
                data_low_u = time_downsample_fixed_step_mean(data_low_u, step=self.step)
                data_low_v = time_downsample_fixed_step_mean(data_low_v, step=self.step)

                data_low = torch.stack([data_low_u, data_low_v], dim=-1)  # [2, 32, 32, 2]
                data_pairs.append((data_low, data_high))
        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        return self.data_pairs[idx]
