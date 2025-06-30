import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FFN, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        输入 x 的形状: (batch_size, seq_len, input_dim)
        输出 y 的形状: (batch_size, seq_len, output_dim)
        """
        residual = x
        # x = x.view(-1, x.size(-1))  # 将 x 展平成二维
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # x = x.view(batch_size, seq_len, -1)  # 将 x 恢复成三维
        x = residual + x
        return x

class bertlayer(nn.Module):
    def __init__(self, embeddingdim, hidden_dim, num_head = 1):
        super(bertlayer, self).__init__()
        self.atte_norm = nn.LayerNorm(embeddingdim)
        self.ffn_norm = nn.LayerNorm(embeddingdim)
        self.atte = torch.nn.MultiheadAttention(embed_dim = embeddingdim, num_heads = num_head, dropout = 0.0)
        self.ffn = FFN(embeddingdim, hidden_dim)
    def forward(self, x, x_padding_mask = None):
        residual = x
        x = self.atte_norm(x)
        x, _ = self.atte(x, x, x, key_padding_mask = x_padding_mask)
        x = x + residual
        x = x + self.ffn(self.ffn_norm(x))
        return x

class bert_feature_extraction(nn.Module):
    def __init__(self, embeddingdim, num_layer):
        super(bert_feature_extraction, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(bertlayer(embeddingdim, embeddingdim * 4))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim = -2)

class TwoLayerRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TwoLayerRegressionModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        return output

import math
class self_attention_model(nn.Module):
    def __init__(self, dim_a, dim_b, dim_c, num_layer_a = 2, num_layer_b = 2):
        super(self_attention_model, self).__init__()
        self.layera = bert_feature_extraction(dim_a, num_layer_a)
        self.layerb = bert_feature_extraction(dim_b, num_layer_b)
        self.header = TwoLayerRegressionModel((dim_a + dim_b + dim_c), math.floor((dim_a + dim_b + dim_c) / 2))
        # self.header = nn.Linear(dim_a + dim_b + dim_c, 1)
    def forward(self, a, b, c):
        a = self.layera(a)
        b = self.layerb(b)
        return torch.sigmoid(self.header(torch.cat([a, b, c], dim = -1)))
        # return self.header(torch.cat([a.mean(-2), b.mean(-2), c], dim = -1))