"""
随机选择一个 gpt_data_preprocess.py 生成的动作序列文件，加载其中的数据，并将其渲染为图像。
"""

import torch
import os
import random
import cv2
import numpy as np

from src.GPT.config.config import Config, set_seed
from src.Renderer.model import *
from src.utils.util import decode, decode_parallel

# 将动作序列渲染为图像
# def decode(x, canvas):
    # x = x.view(-1, 10 + 3)  # 解析为笔触参数 (N, 13)
    # stroke = 1 - Decoder(x[:, :10])  # 生成笔触形状 (N, H, W, 1)
    # stroke = stroke.view(-1, config.canvas_size, config.canvas_size, 1)
    
    # color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)  # 笔触颜色
    # stroke = stroke.permute(0, 3, 1, 2)  # (N, 1, H, W)
    # color_stroke = color_stroke.permute(0, 3, 1, 2)  # (N, 3, H, W)
    
    # stroke = stroke.view(-1, 5, 1, config.canvas_size, config.canvas_size)  # 分为 5 个步骤
    # color_stroke = color_stroke.view(-1, 5, 3, config.canvas_size, config.canvas_size)
    
    # intermediate_results = []
    # for i in range(5):  # 每一步更新画布
    #     canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    #     intermediate_results.append(canvas.clone())
    
    # return canvas, intermediate_results

def save_image(tensor, filename):
    """
    保存张量为图像文件。
    """
    img = tensor[0].detach().cpu().numpy().transpose(1, 2, 0)  # 转换为 (H, W, C)
    img = (img * 255).astype(np.uint8)  # 转换为图像格式
    cv2.imwrite(filename, img)

if __name__ == "__main__":
    # 加载配置
    config = Config(is_training=False)  # 推理模式
    # set_seed(config.seed)  # 设置随机种子
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化画布
    canvas = torch.zeros([1, 3, config.canvas_size, config.canvas_size]).to(device)
    
    # 定义 .pt 文件所在目录
    tensor_dir = "./data/tensor_align_celeba"

    # 获取目录中的所有 .pt 文件
    pt_files = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]

    # 随机选择一个 .pt 文件
    random_file = random.choice(pt_files)
    file_path = os.path.join(tensor_dir, random_file)

    data = torch.load(file_path)
    reshaped_data = torch.tensor(data).view(50, 5, config.action_dim).to(device)
    
    # 加载 Renderer 模型
    Decoder = FCN().to(device)
    Decoder.load_state_dict(torch.load(config.renderer_path, map_location=device))
    Decoder.eval()

    # 逐步解码生成的序列
    intermediate_steps = []
    for i in range(50):
        current_sequence = reshaped_data[i:i+1]
        canvas, res = decode(current_sequence, canvas, Decoder, config.canvas_size)
        intermediate_steps.extend(res)
            
    # 保存结果
    os.makedirs(config.output_dir, exist_ok=True)
    save_image(canvas, os.path.join(config.output_dir, "final_canvas.png"))
    for idx, step in enumerate(intermediate_steps):
        save_image(step, os.path.join(config.output_dir, f"intermediate_step_{idx + 1}.png"))

    # # 获取部分数据输出
    # # last_few_data = data[:5]
    # last_few_data = data[-5:]
    
    # # 打印结果
    # print(last_few_data)
