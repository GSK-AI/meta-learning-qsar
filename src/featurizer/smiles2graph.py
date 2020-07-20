#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Featurizer for graph convolutional neural networks"""


from collections import OrderedDict

import numpy as np

from src.featurizer.molgraph import MolGraph


def featurize_df(
    data_df,
    smiles_col,
    output_col=None,
    adj_type="id_adj_mat",
    explicit_H_node=None,
    **kwargs
):
    """Featurize pandas dataframe

    data_df (pd.DataFrame): DataFrame
    smiles_col (str): SMILES column name
    output_col (list): list of output column names
    adj_type (str): Type of adjacency matrix to use (adj_mat, id_adj_mat, norm_id_adj_mat, norm_laplacian_mat)

    """

    smiles = data_df[smiles_col].values
    adj_mat, node_feat = featurize_smiles(smiles, adj_type, explicit_H_node, **kwargs)

    if output_col is None:
        output = None
    else:
        output = data_df[output_col].values.astype(np.float32)

    featurized = {}
    featurized["adj"] = adj_mat
    featurized["feat"] = node_feat
    featurized["y"] = output
    return featurized


def featurize_smiles(
    smiles, adj_type="id_adj_mat", explicit_H_node=None, **kwargs
):
    """Featurize list or array of SMILES

    smiles (list or np.array): SMILES
    adj_type (str): Type of adjacency matrix to use (adj_mat, id_adj_mat, norm_id_adj_mat, norm_laplacian_mat)
    explicit_H (str or None): Levels of explicit hydrogen ("all", "polar", or None)

    """

    adj_mat = []
    node_feat = []

    for s in smiles:
        adj_mat_mol, node_feat_mol = featurize_single_smiles(
            s, adj_type, explicit_H_node, **kwargs
        )
        adj_mat.append(adj_mat_mol)
        node_feat.append(node_feat_mol)

    adj_mat = np.array(adj_mat)
    node_feat = np.array(node_feat)

    return adj_mat, node_feat


def featurize_single_smiles(smiles, adj_type, explicit_H_node=None, **kwargs):
    mol = MolGraph(smiles, explicit_H_node)
    adj_mat = getattr(mol, adj_type)
    node_feat = mol.node_feat
    return adj_mat, node_feat
