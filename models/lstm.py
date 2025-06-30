import torch
import torch.nn as nn

class TwoLayerLSTM(nn.Module):
    def __init__(self, input_size):
        super(TwoLayerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, input_size * 3, num_layers=2, batch_first=True)
        self.fc = nn.Linear(input_size * 3, input_size)

    def forward(self, x):
        # LSTM 的输出
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_time_step = lstm_out[-1, :]
        # 通过全连接层
        out = self.fc(last_time_step)
        return out

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
class lstm_model(nn.Module):
    def __init__(self, dim_a, dim_b, dim_c, num_layer_a = 2, num_layer_b = 2):
        super(lstm_model, self).__init__()
        self.layer_a = TwoLayerLSTM(dim_a)
        self.layer_b = TwoLayerLSTM(dim_b)
        self.header = TwoLayerRegressionModel((dim_a + dim_b + dim_c), math.floor((dim_a + dim_b + dim_c) / 2))
        # self.header = nn.Linear(dim_a + dim_b + dim_c, 1)
    def forward(self, a, b, c):
        a = self.layer_a(a)
        b = self.layer_b(b)
        return torch.sigmoid(self.header(torch.cat([a, b, c], dim = -1)))
        # return self.header(torch.cat([a.mean(-2), b.mean(-2), c], dim = -1))