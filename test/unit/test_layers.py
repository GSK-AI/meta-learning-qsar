import pytest
import torch

from src.layers import recurrent, convolutional


def test_GRU2D_shape():
    in_dim = 16
    hidden_dim = 16
    batch_size = 4
    num_node = 8

    layer = recurrent.GRU2D(in_dim=in_dim, hidden_dim=hidden_dim, bias=True)
    x = torch.randn(batch_size, num_node, in_dim)
    h0 = torch.randn(batch_size, num_node, in_dim)
    h = layer(x, h0)
    assert list(h.shape) == [batch_size, num_node, hidden_dim]


@pytest.mark.parametrize(
    "in_dim, out_dim, batch_size, num_node, n_edge",
    [(16, 32, 4, 8, 1), (16, 32, 4, 8, 4)],
)
def test_GraphConvolution_shape(in_dim, out_dim, batch_size, num_node, n_edge):
    layer = convolutional.GraphConvolution(
        in_dim=in_dim, out_dim=out_dim, n_edge=n_edge, bias=True
    )
    adj = torch.randn(batch_size, num_node, num_node)
    feat = torch.randn(batch_size, num_node, in_dim)
    out = layer(adj, feat)
    assert list(out.shape) == [batch_size, num_node, out_dim]


@pytest.mark.parametrize(
    "in_dim, out_dim, batch_size, num_node, n_edge",
    [(16, 32, 4, 8, 1), (16, 32, 4, 8, 4)],
)
def test_GraphConvolution_shape(in_dim, out_dim, batch_size, num_node, n_edge):
    layer = convolutional.GraphConvolution(
        in_dim=in_dim, out_dim=out_dim, n_edge=n_edge, bias=True
    )
    adj = torch.randn(batch_size, num_node, num_node)
    feat = torch.randn(batch_size, num_node, in_dim)
    out = layer(adj, feat)
    assert list(out.shape) == [batch_size, num_node, out_dim]


@pytest.mark.parametrize(
    "in_dim, out_dim, batch_size, num_node, n_edge",
    [(16, 16, 4, 8, 1), (16, 16, 4, 8, 4)],
)
def test_GatedGraphConvolution_shape(in_dim, out_dim, batch_size, num_node, n_edge):
    layer = convolutional.GatedGraphConvolution(
        in_dim=in_dim, out_dim=out_dim, n_edge=n_edge, bias=True
    )
    adj = torch.randn(batch_size, num_node, num_node)
    feat = torch.randn(batch_size, num_node, in_dim)
    out = layer(adj, feat)
    assert list(out.shape) == [batch_size, num_node, out_dim]


def test_GatedGraphConvolution_exception():
    in_dim = 16
    out_dim = 32
    with pytest.raises(ValueError):
        layer = convolutional.GatedGraphConvolution(in_dim=in_dim, out_dim=out_dim)
