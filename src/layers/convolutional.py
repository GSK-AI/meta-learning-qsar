import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.recurrent import GRU2D


class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_edge: int=1, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_edge, in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_edge, 1, out_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, adj: torch.Tensor, feat: torch.Tensor):
        if len(adj.size()) == 3:
            adj = adj.unsqueeze(1)
        feat = feat.unsqueeze(1)
        output = torch.matmul(adj, feat)
        output = torch.matmul(output, self.weight)
        if self.bias is not None:
            output = output + self.bias
        output = output.sum(dim=1)
        output = F.relu(output)
        return output


class GatedGraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_edge: int=1, bias:bool=True):
        super(GatedGraphConvolution, self).__init__()
        if in_dim != out_dim:
            raise ValueError(in_dim == out_dim, f"Input ({in_dim}) and output "\
                "({out_dim}) must have the same dimension.")
        self.gc = GraphConvolution(
            in_dim=in_dim, out_dim=out_dim, n_edge=n_edge, bias=bias
        )
        self.gru = GRU2D(in_dim=out_dim, hidden_dim=out_dim, bias=bias)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.gc.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, adj: torch.Tensor, h_0: torch.Tensor):
        h = self.gc(adj, h_0)
        output = self.gru(h, h_0)
        return output
