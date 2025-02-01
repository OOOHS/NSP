"""
用于 GPT 模型的推理脚本，能够生成动作序列并将其渲染为图像。
"""

import os
import torch
import numpy as np
import cv2
import argparse

from src.GPT.model import GPTModel
from src.Renderer.model import FCN
from src.GPT.config.config import Config, set_seed
from src.utils.util import decode, decode_parallel, save_image

# ============================================
# 初始化设置
# ============================================

# 加载配置
config = Config(is_training=False)
set_seed(config.seed)

# 解析命令行参数
parser = argparse.ArgumentParser(description='GPT Inference')
parser.add_argument('--is_parallel', default=True, type=bool, help='use parallel decode or not')
parser.add_argument('--output_dir', default='./output', type=str, help='output directory')
parser.add_argument('--model_path', default='./model_checkpoints/GPT/layers16_truncnum250_epoch1600/model_epoch_1600_loss_51.0138.pth', type=str, help='model path')
args = parser.parse_args()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化画布
canvas = torch.zeros([1, 3, config.canvas_size, config.canvas_size], device=device) # (1, 3, 128, 128)

# ============================================
# 加载训练好的 GPT 模型和 Renderer 模型，并将它们设置为推理模式（eval）。
# 1. 加载 GPT 模型：从指定路径加载状态字典，去除多卡训练的前缀（"module."），并恢复模型权重。
# 2. 加载 Renderer 模型：加载指定路径的渲染器状态字典，并恢复权重。
# ============================================

# 加载 GPT 模型
model = GPTModel(config.attr_dim, config.hidden_dim, config.trunc_num, config.action_dim).to(device)

state_dict = torch.load(args.model_path, map_location=device)
if "module." in list(state_dict.keys())[0]:
    state_dict = {k[7:]: v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model.eval()

# 加载 Renderer 模型
Decoder = FCN().to(device)
Decoder.load_state_dict(torch.load(config.renderer_path, map_location=device))
Decoder.eval()

# ============================================
# Inferencer
# ============================================

# 生成动作序列
def generate_sequence(model, start_attributes, seq_len, feature_dim, device):
    model.eval()
    start_attributes = torch.tensor(start_attributes, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        seq = torch.zeros((1, seq_len, feature_dim), device=device)
        for step in range(seq_len):
            outputs = model(start_attributes, seq[:, :step+1])
            seq[:, step] = outputs[:, step]

    return seq.squeeze(0).cpu().numpy()

# ============================================
# 主函数
# ============================================

if __name__ == "__main__":
    start_attributes = torch.tensor([-1, -1, -1, -1, -1, 
                                     -1, -1, -1, -1, -1, 
                                     -1,  1,  1, -1, -1, 
                                     -1,  1, -1,  1,  1, 
                                      1,  1, -1, -1,  1, 
                                     -1, -1,  1, -1, -1, 
                                     -1,  1,  1,  1,  1, 
                                     -1,  1, -1, -1,  1]).unsqueeze(0)

    generated_sequence = generate_sequence(
        model, start_attributes, seq_len=config.trunc_num, feature_dim=config.action_dim, device=device
    ) # (trunc_num, action_dim)

    # 解码生成的序列
    if not args.is_parallel:
        reshaped_sequence = torch.tensor(generated_sequence).view(-1, 5, config.action_dim).to(device) # (trunc_num // 5, 5, action_dim)
        intermediate_steps = []
        for i in range(config.trunc_num // 5): # 串行解码
            current_sequence = reshaped_sequence[i:i+1]
            canvas, res = decode(current_sequence, canvas, Decoder, config.canvas_size)
            intermediate_steps.extend(res)

        os.makedirs(args.output_dir, exist_ok=True)
        save_image(canvas, os.path.join(args.output_dir, "final_canvas.png"))
        for idx, step in enumerate(intermediate_steps):
            save_image(step, os.path.join(args.output_dir, f"generated{idx}.png"))
    else:
        generated_sequence = torch.tensor(generated_sequence).unsqueeze(0).to(device) # (1, trunc_num, action_dim)
        canvas = decode_parallel(generated_sequence, canvas, Decoder, config.canvas_size) # 并行解码
        
        os.makedirs(args.output_dir, exist_ok=True)
        save_image(canvas, os.path.join(args.output_dir, "final_canvas.png"))
        
    print(f"生成完成，图像已保存到 {args.output_dir} 文件夹中。")
