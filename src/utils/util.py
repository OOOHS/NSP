import os
import torch
from torch.autograd import Variable
import numpy as np

USE_CUDA = torch.cuda.is_available()

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, device):
    return torch.tensor(ndarray, dtype=torch.float, device=device)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for m1, m2 in zip(target.modules(), source.modules()):
        m1._buffers = m2._buffers.copy()
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir

import cv2
# 保存图像
def save_image(tensor, filename):
    img = tensor[0].detach().cpu().numpy().transpose(1, 2, 0) # 转换为 (H, W, C)
    img = (img * 255).astype(np.uint8) # 转换为图像格式
    cv2.imwrite(filename, img) # 保存图像

# 串行解码
def decode(x, canvas, Decoder, width):  
    # 确定目标设备（Decoder 的设备）
    device = next(Decoder.parameters()).device
    
    # 将所有张量移动到目标设备
    x = x.view(-1, 10 + 3).to(device)  # 确保 x 在目标设备上
    canvas = canvas.to(device)         # 确保 canvas 在目标设备上

    # 解码生成 stroke 和颜色
    stroke = 1 - Decoder(x[:, :10])    # (b, width * width)
    stroke = stroke.view(-1, width, width, 1)  # (b, width, width, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)  # (b, width, width, 3)
    
    # 调整维度
    stroke = stroke.permute(0, 3, 1, 2)           # (b, 1, width, width)
    color_stroke = color_stroke.permute(0, 3, 1, 2)  # (b, 3, width, width)
    
    # 拆分批次中的 stroke 和颜色
    stroke = stroke.view(-1, 5, 1, width, width)       # (b, 5, 1, width, width)
    color_stroke = color_stroke.view(-1, 5, 3, width, width)  # (b, 5, 3, width, width)

    res = []  # 存储中间结果
    for i in range(5):
        # 确保所有操作张量都在同一设备
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas.clone())  # 记录每一步的中间结果
    
    return canvas, res

# 数据预处理
def preprocess_attributes(attr_tensor):
    # 生成与 attr_tensor 形状相同的标准正态分布噪声
    noise = torch.randn_like(attr_tensor)
    # 拼接属性张量和噪声张量
    return torch.cat((attr_tensor, noise), dim=-1)

# # 串行解码
# def decode(x, canvas, Decoder, width):  # b * (10 + 3)
#     x = x.view(-1, 10 + 3) # (b, 13)
#     stroke = 1 - Decoder(x[:, :10]) # (b, width * width)
#     stroke = stroke.view(-1, width, width, 1)  # (b, width, width, 1)
#     color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
#     stroke = stroke.permute(0, 3, 1, 2)
#     color_stroke = color_stroke.permute(0, 3, 1, 2)
#     stroke = stroke.view(-1, 5, 1, width, width)
#     color_stroke = color_stroke.view(-1, 5, 3, width, width)
#     res = []
#     for i in range(5):
#         canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
#         res.append(canvas)
#     return canvas, res

# 并行解码
def decode_parallel(x, canvas, Decoder, width):
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    action_dim = x.shape[2]

    # 展平时间维度
    x = x.view(-1, action_dim)

    # 计算笔画的不透明度
    stroke = 1 - Decoder(x[:, :10])
    stroke = stroke.view(batch_size, seq_len, width, width, 1)

    # 准备颜色信息
    color = x[:, -3:].view(batch_size, seq_len, 1, 1, 3)
    color_stroke = stroke * color

    # 调整维度顺序
    stroke = stroke.permute(0, 1, 4, 2, 3)
    color_stroke = color_stroke.permute(0, 1, 4, 2, 3)

    # 计算(1-stroke)
    transparency = 1 - stroke
    
    # 计算后缀积(从后向前的累积效果)
    # 使用flip和cumprod来实现
    reversed_transparency = transparency.flip(dims=[1])
    cumulative_reversed = torch.cumprod(reversed_transparency, dim=1)
    future_transparency = torch.cat([
        cumulative_reversed.flip(dims=[1])[:, 1:],
        torch.ones_like(transparency[:, :1])
    ], dim=1)

    # 计算每个color_stroke的最终贡献
    final_contribution = color_stroke * future_transparency
    
    # 累加所有贡献
    final_canvas = final_contribution.sum(dim=1)

    return final_canvas
