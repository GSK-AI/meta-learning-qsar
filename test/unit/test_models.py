import pytest
import torch
from src.models import ggnn, l2l_maml
import copy


def _setup_ggnn():
    batch_size = 4
    num_node = 8
    in_dim = 4
    n_conv = 2
    n_edge = 1
    fc_dims = [8, 1]

    adj = torch.randn(batch_size, num_node, num_node)
    feat = torch.randn(batch_size, num_node, in_dim)
    atom_vec = torch.ones(batch_size, num_node, 1)
    x = [adj, feat, atom_vec]
    model = ggnn.GatedGraphNeuralNetwork(
        in_dim=in_dim, n_conv=n_conv, fc_dims=fc_dims, n_edge=n_edge
    )
    return model, x, batch_size, fc_dims


def test_ggnn_shape():
    model, x, batch_size, fc_dims = _setup_ggnn()
    output = model(x)
    assert list(output.shape) == [batch_size, fc_dims[-1]]


def test_ggnn_gradients():
    """Test that every parameters is updated"""
    model, x, batch_size, fc_dims = _setup_ggnn()
    original_model = copy.deepcopy(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    output = model(x)
    loss = torch.sum(1 - output)
    loss.backward()
    optimizer.step()
    for p_original, p_step in zip(original_model.parameters(), model.parameters()):
        assert all(torch.flatten(torch.ne(p_original, p_step)).tolist())


def test_ggnn_transfer():
    """Test that output layer is reinitialized"""
    model, x, batch_size, fc_dims = _setup_ggnn()
    original_layer = copy.deepcopy(model.fc_layers[-1])
    model.transfer(out_dim=1)
    new_layer = model.fc_layers[-1]
    assert all(torch.flatten(torch.ne(original_layer.weight, new_layer.weight)).tolist())


def test_l2l_maml_anil():
    """Test that MAML only updates output layer parameters in ANIL mode"""
    model, x, batch_size, fc_dims = _setup_ggnn()
    original_model = copy.deepcopy(model)
    learner = l2l_maml.MAML(model, lr=0.001, first_order=False, anil=True)
    output = learner(x)
    loss = torch.sum(1 - output)
    learner.adapt(loss)
    
    original_params = list(original_model.parameters())
    new_params = list(learner.parameters())
    for i, (p_original, p_step) in enumerate(zip(original_params, new_params)):
        if i < (len(original_params) - 2):
            assert all(torch.flatten(torch.eq(p_original, p_step)).tolist())
        else:
            assert all(torch.flatten(torch.ne(p_original, p_step)).tolist())    
