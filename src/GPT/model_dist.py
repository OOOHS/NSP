"""
相比model.py，该实现进行了一些改进：
1.添加了sample方法。
2.去除了模型本身的归一化，只对采样结果进行截断。
待完成：
3.实现 KV cache。
4.输出高斯分布的参数，而不是直接输出预测值。-感觉有点玄学不知道有没有用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical

class CausalSelfAttention(nn.Module):
    """因果自注意力机制(带KV cache)"""
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** 0.5
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None, kv_cache=None, layer_idx=None):
        batch_size, seq_len, hidden_dim = x.size()
        
        # 计算当前输入的 Q,K,V
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用 KV cache
        if kv_cache is not None:
            if layer_idx in kv_cache:
                past_k, past_v = kv_cache[layer_idx]
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            kv_cache[layer_idx] = (k, v)
            
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == float('-inf'), float('-inf'))
            
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.out_proj(context)

class TransformerBlock(nn.Module):
    """单个 Transformer Block"""
    def __init__(self, hidden_dim, num_heads, intermediate_size):
        super().__init__()
        self.attention = CausalSelfAttention(hidden_dim, num_heads)
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_dim),
            nn.Dropout(0.1),
        )
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-5)

    def forward(self, x, mask=None, kv_cache=None, layer_idx=None):
        attn_output = self.attention(x, mask, kv_cache, layer_idx)
        x = self.ln_1(x + attn_output)  # 残差连接
        feed_forward_output = self.mlp(x)
        return self.ln_2(x + feed_forward_output)  # 残差连接

# GPT 模型 Backbone
class GPTModelBB(nn.Module):
    def __init__(self, attr_dim, hidden_dim, seq_len, action_dim, num_heads, num_layers):
        super().__init__()
        self.attr_dim = attr_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 输入的全连接层
        self.fc_attr = nn.Linear(attr_dim, hidden_dim)
        self.fc_seq = nn.Linear(action_dim, hidden_dim)

        # 位置编码：learnable embedding
        self.pos_embedding = nn.Embedding(self.seq_len + 1, hidden_dim)

        # Transformer 层
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, hidden_dim * 4) for _ in range(num_layers)]
        )

    def generate_square_subsequent_mask(self, sz):
        """生成一个因果mask"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, attributes, sequence):
        # 将属性和序列投影到 hidden_dim 维度
        attr_emb = self.fc_attr(attributes)  # (batch_size, 1, hidden_dim)
        seq_emb = self.fc_seq(sequence)      # (batch_size, seq_len, hidden_dim)

        # 拼接属性和序列
        input_emb = torch.cat((attr_emb, seq_emb), dim=1)  # (batch_size, seq_len+1, hidden_dim)

        # 加入位置编码
        seq_len_plus_1 = input_emb.size(1)
        positions = torch.arange(0, seq_len_plus_1, device=input_emb.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        input_emb = input_emb + pos_emb

        # 获取因果mask
        mask = self.generate_square_subsequent_mask(seq_len_plus_1).to(input_emb.device)

        # 应用 Transformer 层
        hidden_states = input_emb
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)

        return hidden_states # (batch_size, seq_len+1, hidden_dim)

# 高斯混合分布参数预测层
class ParameterPredictionLayer(nn.Module):
    def __init__(self, hidden_dim, gaussian_num, action_dim, scale):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gaussian_num = gaussian_num
        self.action_dim = action_dim
        self.scale = scale

        # 输出高斯混合分布参数
        self.pi_out = nn.Linear(hidden_dim, gaussian_num)  # 混合权重
        self.mean_out = nn.Linear(hidden_dim, gaussian_num * action_dim)  # 高斯均值
        
        # 输出 Cholesky 分解的下三角矩阵
        self.chol_out = nn.Linear(hidden_dim, gaussian_num * (action_dim * (action_dim + 1) // 2))

        # 初始化一个较小的正值，用于对角线元素
        self.diag_bias = nn.Parameter(torch.ones(gaussian_num, action_dim) * 0.1)

    def forward(self, hidden_states):
        # 获取生成序列的输出部分
        output_seq = hidden_states[:, :-1, :]  # 最后一个时间步不需要

        # 输出高斯混合分布的参数
        pi = self.pi_out(output_seq)  # (batch_size, seq_len, gaussian_num)
        pi = F.softmax(pi, dim=-1)  # 在每个时间步上应用 softmax

        mu = self.mean_out(output_seq)  # (batch_size, seq_len, gaussian_num * action_dim)
        mu = mu.view(output_seq.size(0), output_seq.size(1), self.gaussian_num, self.action_dim)

        # 计算 Cholesky 分解的下三角矩阵
        chol = self.chol_out(output_seq)  # (batch_size, seq_len, gaussian_num * (action_dim * (action_dim + 1) // 2))
        chol = chol.view(output_seq.size(0), output_seq.size(1), self.gaussian_num, -1)
        
        # 构建完整的 Cholesky 下三角矩阵
        tril_indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        L = torch.zeros(
            output_seq.size(0), output_seq.size(1), self.gaussian_num, self.action_dim, self.action_dim,
            device=hidden_states.device, dtype=chol.dtype
        )
        
        # 填充非对角线元素，使用 tanh 进行约束
        off_diag_mask = tril_indices[0] != tril_indices[1]
        L[:, :, :, tril_indices[0][off_diag_mask], tril_indices[1][off_diag_mask]] = torch.tanh(chol[:, :, :, off_diag_mask]) * 0.1

        # 填充对角线元素，确保为正且不太小
        diag_idx = torch.arange(self.action_dim)
        diag_values = F.softplus(chol[:, :, :, :self.action_dim] + self.diag_bias[None, None, :, :])
        L[:, :, :, diag_idx, diag_idx] = diag_values.to(L.dtype) + 0.1  # 添加一个小的正值以确保正定性

        # 计算协方差矩阵
        full_cov = torch.matmul(L, L.transpose(-1, -2))

        # 添加一个小的对角矩阵以确保数值稳定性
        full_cov = full_cov + torch.eye(self.action_dim, device=full_cov.device, dtype=full_cov.dtype) * 1e-5

        # 应用缩放因子
        full_cov = full_cov

        return pi, mu, full_cov

# GPT 模型
class GPTModel(nn.Module):
    def __init__(self, attr_dim, hidden_dim, seq_len, action_dim, num_heads, num_layers, scale, gaussian_num):
        super().__init__()
        self.transformer = GPTModelBB(attr_dim, hidden_dim, seq_len, action_dim, num_heads, num_layers)
        self.param_predictor = ParameterPredictionLayer(hidden_dim, gaussian_num, action_dim, scale)

    def forward(self, attributes, sequence):
        # 通过 Transformer 骨架生成隐藏状态
        hidden_states = self.transformer(attributes, sequence) # (batch_size, seq_len+1, hidden_dim)
        # 通过参数预测层生成高斯分布参数
        pi, mu, full_cov = self.param_predictor(hidden_states) # 不需要最后一个时间步的输出
        return pi, mu, full_cov 

# 条件属性增强器
class ConditionalAttributeEnhancer(nn.Module):
    def __init__(self, attr_dim, hidden_dim=128):
        super().__init__()
        self.attr_dim = attr_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值预测器
        self.mean_predictor = nn.Linear(hidden_dim, attr_dim)
        
        # 方差预测器（输出对数方差）
        self.logvar_predictor = nn.Linear(hidden_dim, attr_dim)
        
    def forward(self, attr):
        # 编码属性
        h = self.encoder(attr)
        
        # 预测条件均值和对数方差
        mean = self.mean_predictor(h)
        logvar = self.logvar_predictor(h)
        
        # 将对数方差转换为方差
        var = torch.exp(logvar)
        
        # 使用重参数化技巧进行采样
        if self.training:
            epsilon = torch.randn_like(mean)
            sample = mean + torch.sqrt(var) * epsilon
        else:
            sample = mean  # 在推理时，直接使用均值作为样本
        
        return sample, mean, var

# 使用了条件属性增强器的 GPT 模型
class GPTModelWithCAE(nn.Module):
    def __init__(self, attr_dim, hidden_dim, seq_len, action_dim, num_heads, num_layers, scale, gaussian_num):
        super().__init__()
        self.transformer = GPTModelBB(attr_dim, hidden_dim, seq_len, action_dim, num_heads, num_layers)
        self.param_predictor = ParameterPredictionLayer(hidden_dim, gaussian_num, action_dim, scale)
        self.attr_enhancer = ConditionalAttributeEnhancer(attr_dim, hidden_dim)

    def forward(self, attributes, sequence):
        # 通过属性增强器增强属性
        enhanced_attr, mean, var = self.attr_enhancer(attributes)
        
        # 通过 Transformer 骨架生成隐藏状态
        hidden_states = self.transformer(enhanced_attr, sequence)
        
        # 通过参数预测层生成高斯分布参数
        pi, mu, full_cov = self.param_predictor(hidden_states)
        
        return pi, mu, full_cov, mean, var


class MDNLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(MDNLoss, self).__init__()
        self.eps = eps

    def forward(self, pi, mu, full_cov, y):
        batch_size, seq_len, gaussian_num, action_dim = mu.shape

        # 将y扩展到与mu相同的形状
        y = y.unsqueeze(2).expand(-1, -1, gaussian_num, -1)

        # 创建多元高斯分布
        mu = mu.float() # 我也不知道为什么要转成float，但是我也不敢不转
        full_cov = full_cov.float()
        mvn = MultivariateNormal(loc=mu, covariance_matrix=full_cov)

        # 计算每个高斯分量的对数概率密度
        log_probs = mvn.log_prob(y)  # (batch_size, seq_len, gaussian_num)

        # 创建分类分布（用于混合权重）
        cat = Categorical(probs=pi)

        # 计算混合分布的对数概率
        log_prob = torch.logsumexp(cat.logits + log_probs, dim=2)

        # 计算负对数似然损失
        loss = -torch.mean(log_prob)

        return loss