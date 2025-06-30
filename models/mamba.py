import torch
import torch.nn as nn
from mamba_ssm import Mamba
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

class mambalayer(nn.Module):
    def __init__(self, embeddingdim, hidden_dim):
        super(mambalayer, self).__init__()
        self.atte_norm = nn.LayerNorm(embeddingdim)
        self.ffn_norm = nn.LayerNorm(embeddingdim)
        self.atte = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model= embeddingdim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.ffn = FFN(embeddingdim, hidden_dim)
    def forward(self, x):
        residual = x
        x = self.atte_norm(x)
        x = self.atte(x.unsqueeze(0)).squeeze(0)
        x = x + residual
        x = x + self.ffn(self.ffn_norm(x))
        return x

class mamba_feature_extraction(nn.Module):
    def __init__(self, embeddingdim, num_layer):
        super(mamba_feature_extraction, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(mambalayer(embeddingdim, embeddingdim * 4))
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
class mamba_model(nn.Module):
    def __init__(self, dim_a, dim_b, dim_c, num_layer_a = 2, num_layer_b = 2):
        super(mamba_model, self).__init__()
        self.layera = mamba_feature_extraction(dim_a, num_layer_a)
        self.layerb = mamba_feature_extraction(dim_b, num_layer_b)
        self.header = TwoLayerRegressionModel((dim_a + dim_b + dim_c), math.floor((dim_a + dim_b + dim_c) / 2))
        # self.header = nn.Linear(dim_a + dim_b + dim_c, 1)
    def forward(self, a, b, c):
        a = self.layera(a)
        b = self.layerb(b)
        return torch.sigmoid(self.header(torch.cat([a, b, c], dim = -1)))
        # return self.header(torch.cat([a.mean(-2), b.mean(-2), c], dim = -1))