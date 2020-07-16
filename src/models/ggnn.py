#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Graph Convolutional Neural Network for Pretraining

__author__ = Cuong Nguyen
__email__ = "cuong.q.nguyen@gsk.com"

"""


import math
from typing import Any, List, Set, Dict, Tuple, Optional, Iterable, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.convolutional import GatedGraphConvolution

warnings.simplefilter("ignore")


class GatedGraphNeuralNetwork(nn.Module):
    """A variant of the graph neural network family that utilizes GRUs
    to control the flow of information between layers.

    Each GatedGraphConvolution operation follows the formulations below:

    .. math:: H^{(L_i)} = A H^{(L-1)} W^{(L)}
    .. math:: H^{(L)} = GRU(H^{(L-1)}, H^{(L_i)})

    The current implementation also facilites transfer learning with the "transfer" method.
    The method replaces the last fully connected layer in the trained model object
    with a reinitialized layer that has a specified output dimension.

    Example:
    >>> # here we instantiate a model with output dimension 1
    >>> model = GatedGraphNeuralNetwork(n_edge=1, in_dim=10, n_conv=5, fc_dims=[1024, 1])
    >>> # now we reinitializes the last layer to have output dimension of 50
    >>> model.transfer(out_dim=50)

    Gated Graph Sequence Neural Networks: https://arxiv.org/abs/1511.05493
    Neural Message Passing for Quantum Chemistry: https://arxiv.org/abs/1704.01212

    """
    def __init__(
        self,
        n_edge: int,
        in_dim: int,
        n_conv: int,
        fc_dims: Iterable[int],
        p_dropout: float = 0.2,
    ) -> None:
        """Gated graph neural network with support for transfer learning

        Parameters
        ----------
        n_edge : int
            Number of edges in input graphs.
        in_dim : int
            Number of features per node in input graphs.
        n_conv : int
            Number of gated graph convolution layers.
        fc_dims : Iterable[int]
            Fully connected layers dimensions.
        """
        super(GatedGraphNeuralNetwork, self).__init__()

        self.conv_layers, self.fc_layers = self._build_layers(
            in_dim=in_dim, n_edge=n_edge, fc_dims=fc_dims, n_conv=n_conv
        )

        self.dropout = nn.Dropout(p=p_dropout)
        self.reset_parameters()

    @staticmethod
    def _build_layers(in_dim, n_edge, fc_dims, n_conv):
        conv_layers = []

        for i in range(n_conv):
            l = GatedGraphConvolution(in_dim=in_dim, out_dim=in_dim, n_edge=n_edge)
            conv_layers.append(l)

        fc_layers = []
        num_fc_layers = len(fc_dims)
        fc_dims.insert(0, in_dim)
        for i, (in_dim, out_dim) in enumerate(zip(fc_dims[:-1], fc_dims[1:])):
            l = nn.Linear(in_dim, out_dim)

            if i < (num_fc_layers - 2):
                l = nn.Sequential(l, nn.ReLU())
            elif i == (num_fc_layers - 2):
                l = nn.Sequential(l, nn.Tanh())

            fc_layers.append(l)

        return nn.ModuleList(conv_layers), nn.ModuleList(fc_layers)

    def reset_parameters(self):
        for l in self.conv_layers:
            l.reset_parameters()

        for k, v in self.state_dict().items():
            if "fc_layers" in k:
                if "weight" in k:
                    nn.init.xavier_uniform_(v)
                elif "bias" in k:
                    nn.init.zeros_(v)

    def encode(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Encode featurized batched input.
        This is done by forward propagating up to the second to last layer in the network.

        Parameters
        ----------
        x : List[torch.Tensor]
            List of batch input torch.Tensor [adj_mat, node_feat, atom_vec ]

        Returns
        -------
        torch.Tensor
            Encoded inputs
        """
        adj, node_feat, atom_vec = x

        for layer in self.conv_layers:
            node_feat = layer(adj, node_feat)
            node_feat = self.dropout(node_feat)
        output = torch.mul(node_feat, atom_vec)

        output = output.sum(1)

        for layer in self.fc_layers[:-1]:
            output = layer(output)
            output = self.dropout(output)

        return output

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Run forward pass on batched input.

        Parameters
        ----------
        x : List[torch.Tensor]
            List of batch input torch.Tensor [adj_mat, node_feat, atom_vec]

        Returns
        -------
        torch.Tensor
            Model output
        """
        output = self.encode(x)
        output = self.fc_layers[-1](output)
        return output

    def transfer(self, out_dim: Union[list, int], freeze: bool = False) -> None:
        """Replace the last fully connected layer with a newly initialized layer
        with out_dim as output dimension. Use freeze=True to freeze the pre-trained
        network and use it as a featurizer.

        Parameters
        ----------
        out_dim : Union[list,int]
            Output dimension of the new fully connected layer
        freeze : bool, optional
            Freeze the weights of the pretrained network, by default False
        """
        # only transfer learn on graph level
        self.dropout = nn.Dropout(p=0.1)

        # freeze parameters if necessary
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        # new final fc layer overriding freeze
        if isinstance(out_dim, int):
            out_dim = [out_dim]

        in_dim = self.fc_layers[-1].in_features
        out_dim.insert(0, in_dim)
        del self.fc_layers[-1]

        for i, (in_dim, out_dim_) in enumerate(zip(out_dim[:-1], out_dim[1:])):
            layer = nn.Linear(in_dim, out_dim_)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.fc_layers.append(layer)

        return None
