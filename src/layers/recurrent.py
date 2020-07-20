import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU2D(nn.Module):
    """2D GRU Cell"""
    def __init__(self, in_dim, hidden_dim, bias=True):
        super(GRU2D, self).__init__()
        self.x_to_intermediate = nn.Linear(in_dim, 3 * hidden_dim, bias=bias)
        self.h_to_intermediate = nn.Linear(in_dim, 3 * hidden_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self.state_dict().items():
            if "weight" in k:
                std = math.sqrt(6.0 / (v.size(1) + v.size(0) / 3))
                nn.init.uniform_(v, a=-std, b=std)
            elif "bias" in k:
                nn.init.zeros_(v)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor):
        intermediate_x = self.x_to_intermediate(x)
        intermediate_h = self.h_to_intermediate(h_0)

        x_r, x_z, x_n = intermediate_x.chunk(3, -1)
        h_r, h_z, h_n = intermediate_h.chunk(3, -1)

        r = torch.sigmoid(x_r + h_r)
        z = torch.sigmoid(x_z + h_z)
        n = torch.tanh(x_n + (r * h_n))
        h = (1 - z) * n + z * h_0

        return h