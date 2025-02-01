import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    """因果自注意力机制"""
    def __init__(self, hidden_dim, num_heads):
        super(CausalSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        batch_size, seq_len, hidden_dim = x.size()
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 因果 mask
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(1) == float('-inf'), float('-inf'))

        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.out_proj(context)

class TransformerBlock(nn.Module):
    """单个 Transformer Block"""
    def __init__(self, hidden_dim, num_heads, intermediate_size):
        super(TransformerBlock, self).__init__()
        self.attention = CausalSelfAttention(hidden_dim, num_heads)
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_dim),
            nn.Dropout(0.1),
        )
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-5)

    def forward(self, x, mask):
        attn_output = self.attention(x, mask)
        x = self.ln_1(x + attn_output)  # 残差连接
        feed_forward_output = self.mlp(x)
        return self.ln_2(x + feed_forward_output)  # 残差连接

class GPTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, out_dim, num_heads=16, num_layers=16):
        super(GPTModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers  # Transformer层数

        self.fc_attr = nn.Linear(input_dim, hidden_dim)
        self.fc_seq = nn.Linear(out_dim, hidden_dim)

        # 位置编码：learnable embedding
        # 注意这里的 +1 是因为会将属性拼到序列前面
        self.pos_embedding = nn.Embedding(self.seq_len + 1, hidden_dim)

        # 自定义 Transformer 层
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, hidden_dim * 4) for _ in range(num_layers)]
        )
        # self.ln_f = nn.LayerNorm(hidden_dim, eps=1e-5)

        # 输出层
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def generate_square_subsequent_mask(self, sz):
        """生成一个因果mask，确保每个位置只能看到之前的位置"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))  # 上三角部分为负无穷
        return mask

    def forward(self, attributes, sequence):
        # 将属性和序列投影到 hidden_dim 维度
        attr_emb = self.fc_attr(attributes)  # (batch_size, 1, hidden_dim)
        seq_emb = self.fc_seq(sequence)       # (batch_size, seq_len, hidden_dim)

        # 拼接属性和序列
        input_emb = torch.cat((attr_emb, seq_emb), dim=1)  # (batch_size, seq_len+1, hidden_dim)

        # 加入位置编码
        seq_len_plus_1 = input_emb.size(1)
        positions = torch.arange(0, seq_len_plus_1, device=input_emb.device).unsqueeze(0)  # shape: (1, seq_len+1)
        pos_emb = self.pos_embedding(positions)  # shape: (1, seq_len+1, hidden_dim)
        input_emb = input_emb + pos_emb

        # 获取因果mask
        mask = self.generate_square_subsequent_mask(seq_len_plus_1).to(input_emb.device)

        # 应用自定义 Transformer 层
        hidden_states = input_emb
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)

        # # 最后一个 LayerNorm（如需可取消注释）
        # hidden_states = self.ln_f(hidden_states)

        # 获取生成序列的输出部分
        output_seq = hidden_states[:, :-1, :]  # 左移最后一个位置

        # 输出序列预测
        pred = self.fc_out(output_seq)

        # 添加归一化，限制输出在 [0, 1] 之间
        # pred = torch.sigmoid(pred)
        pred = 0.5 * (torch.tanh(pred) + 1)

        return pred