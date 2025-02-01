"""
用于 GPT 模型的推理脚本，能够生成动作序列并将其渲染为图像。
"""

import os
import torch
import numpy as np
import cv2
import argparse

from src.GPT.model_dist import GPTModelWithCAE
from src.Renderer.model import FCN
from src.GPT.config.config import Config, set_seed
from src.utils.util import decode, decode_parallel, save_image

# ============================================
# 初始化设置
# ============================================

# 加载配置
config = Config(is_training=False)
# set_seed(config.seed)

# 解析命令行参数
parser = argparse.ArgumentParser(description='GPT Inference')
parser.add_argument('--is_parallel', default=False, type=bool, help='use parallel decode or not')
parser.add_argument('--output_dir', default='./output', type=str, help='output directory')
parser.add_argument('--model_path', default='./model_checkpoints/model_epoch_10_loss_21.1271.pth', type=str, help='model path')
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
model = GPTModelWithCAE(**config.get_gpt_config()).to(device)

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


from torch.distributions import MultivariateNormal, Categorical
def generate_sequence(model, start_attributes, seq_len, action_dim, device):
    model.eval()
    start_attributes = torch.tensor(start_attributes, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        seq = torch.zeros((1, seq_len, action_dim), device=device)
        for step in range(seq_len):
            pi, mu, full_cov, _, _ = model(start_attributes, seq[:, :step+1])
            # pi: (batch_size, seq_len, gaussian_num)
            # mu: (batch_size, seq_len, gaussian_num, action_dim)
            # full_cov: (batch_size, seq_len, gaussian_num, action_dim, action_dim)
            
            cur_pi = pi[:, -1, :]  # (batch_size, gaussian_num)
            cur_mu = mu[:, -1, :, :]  # (batch_size, gaussian_num, action_dim)
            cur_full_cov = full_cov[:, -1, :, :, :]  # (batch_size, gaussian_num, action_dim, action_dim)
            
            batch_size, gaussian_num = cur_pi.size()
            
            # 采样
            # 利用 cur_pi 选择高斯分布的索引
            cat = Categorical(probs=cur_pi)
            sampled_indices = cat.sample()  # (batch_size,)
            
            # 利用采样的索引从 mu 和 full_cov 中取值
            batch_indices = torch.arange(batch_size, device=device)
            cur_mu = cur_mu[batch_indices, sampled_indices]  # (batch_size, action_dim)
            cur_full_cov = cur_full_cov[batch_indices, sampled_indices]  # (batch_size, action_dim, action_dim)
            
            
            # 从多元高斯分布中采样
            mvn = MultivariateNormal(loc=cur_mu, covariance_matrix=cur_full_cov)
            cur_action = mvn.sample()  # (batch_size, action_dim)
            
            # 保存采样的动作
            seq[:, step] = cur_action # (1, seq_len, action_dim)
    
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
    # start_attributes = preprocess_attributes_inf(start_attributes)

    generated_sequence = generate_sequence(
        model, start_attributes, seq_len=config.trunc_num, action_dim=config.action_dim, device=device
    ) # (trunc_num, action_dim)
    
    # 缩放回原始范围
    generated_sequence = generated_sequence / config.scale
    generated_sequence = torch.tensor(generated_sequence).clamp(0, 1) # 限制在 [0, 1] 范围内

    # 解码生成的序列
    if not args.is_parallel:
        reshaped_sequence = generated_sequence.view(-1, 5, config.action_dim).to(device) # (trunc_num // 5, 5, action_dim)
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
        generated_sequence = generated_sequence.unsqueeze(0).to(device) # (1, trunc_num, action_dim)
        canvas = decode_parallel(generated_sequence, canvas, Decoder, config.canvas_size) # 并行解码
        
        os.makedirs(args.output_dir, exist_ok=True)
        save_image(canvas, os.path.join(args.output_dir, "final_canvas.png"))
        
    print(f"生成完成，图像已保存到 {args.output_dir} 文件夹中。")
