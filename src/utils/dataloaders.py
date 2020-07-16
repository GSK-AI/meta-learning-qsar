import logging
import os
import pickle
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, _utils

from src.utils.dataset import GraphDataset

default_collate = _utils.collate.default_collate


def get_loaders(source_path: str, inner_batch_size: int = 64) -> dict:
    """Create dataloaders for meta learning

    source_path should contain three files (see train_maml documentation for details)
    - featurized_data.pkl
    - meta_split_idx.pkl
    - dataset_split_idx.pkl

    This function will return a dictionary of dataloaders with schema. The two files
    meta_split_idx.pkl and dataset_split_idx.pkl determines which the assignment of each loader.
        {
            "meta_train": {
                "train": [loader_0, loader_1...],
                "val": [loader_0, loader_1...],
                "test": [loader_0, loader_1...]
            },
            "meta_val": {
                ...
            },
            "meta_train": {
                ...
            }
        }

    Parameters
    ----------
    source_path : str
        Path to data source directory
    inner_batch_size : int, optional
        Batch size for inner loop updates, by default 64

    Returns
    -------
    dict
        Instantiated dataloaders for meta learning
    """
    data = {}
    for file in ["featurized_data.pkl", "meta_split_idx.pkl", "dataset_split_idx.pkl"]:
        try:
            with open(os.path.join(source_path, file), "rb") as f:
                data[file] = pickle.load(f)
        except FileNotFoundError as e:
            logging.error(f"File {file} not found in provided source_path {source_path}")
            sys.exit(1)

    loaders = create_task_data(
        data["featurized_data.pkl"],
        data["meta_split_idx.pkl"],
        data["dataset_split_idx.pkl"],
        inner_batch_size,
    )

    return loaders


def create_task_data(
    data: dict, meta_split_idx: dict, dataset_split_idx: dict, inner_batch_size: int
) -> dict:
    """Create task dataloaders

    Parameters
    ----------
    data : dict
        Featurized molecular graphs data with keys ["adj", "feat"]
    meta_split_idx : dict
        Task split indices
    dataset_split_idx : dict
        Instance split indices
    inner_batch_size : int
        Batch size to create dataloader

    Returns
    -------
    dict
        Instantiated dataloaders for meta learning
    """
    if isinstance(meta_split_idx, dict):
        loaders = {}
        for k, v in meta_split_idx.items():
            loaders[k] = create_task_data(data, v, dataset_split_idx[k], inner_batch_size)

    elif isinstance(meta_split_idx, list):
        loaders = []
        for i, task_idx in enumerate(meta_split_idx):
            dataset_split_idx_task = {
                k: dataset_split_idx[k][i] for k in ["train", "val", "test"]
            }
            loaders.append(
                create_task_data(data, task_idx, dataset_split_idx_task, inner_batch_size)
            )
        loaders = {k: [l[k] for l in loaders] for k in loaders[0]}

    elif isinstance(meta_split_idx, int):
        loaders = {}
        y = data["y"][:, meta_split_idx]
        # Filter by np.nan if nan is found, else filter by -1
        if any(np.isnan(y)):
            mask = ~np.isnan(y)
        else:
            mask = y != -1
        y = y[mask]
        adj = data["adj"][mask]
        feat = data["feat"][mask]

        for k, v in dataset_split_idx.items():
            loaders[k] = build_dataloader(
                x=[adj[v], feat[v]],
                y=y[v],
                batch_size=inner_batch_size,
                shuffle=True if "train" in k else False,
                num_workers=0,
            )

    return loaders


def build_dataloader(
    x: List[np.ndarray],
    batch_size: int,
    y: np.ndarray = None,
    shuffle: bool = False,
    num_workers: int = 2,
) -> torch.utils.data.DataLoader:
    """Create pytorch dataloader

    Parameters
    ----------
    x : List[np.ndarray]
        List of [adjacency, node features]
    batch_size : int
        Batch size
    y : np.ndarray, optional
        Labels for training, by default None
    shuffle : bool, optional
        Shuffle data, by default False
    num_workers : int, optional
        Number of workers for DataLoader, by default 2

    Returns
    -------
    torch.utils.data.DataLoader
        Instantiated DataLoader
    """
    dataset = GraphDataset(x=x, y=y)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=post_batch_padding_collate_fn,
    )

    return dataloader


def post_batch_padding_collate_fn(batch_data: List[tuple]) -> Tuple[torch.Tensor]:
    """Custom collate function for PyTorch DataLoader.
    This function is used to zero pad inputs to the same dimentions after batching.
    
    Arguments:
        batch_data {list} -- list of tuple (x, y) from GraphDataset
            - x: list of torch tensors [adj_mat, node_feat, atom_vec]
            - y: torch tensor

    Parameters
    ----------
    batch_data : List[tuple]
        List of tuples (x, y) with length of batch size.
        Each element in the list is an instance from GraphDataset.
            - x: List[torch.Tensor]
            - y: torch.Tensor

    Returns
    -------
    Tuple[torch.Tensor]
        Tuple (x, y) of padded and batched tensors.
    """
    x, y = zip(*batch_data)
    adj_mat, node_feat, atom_vec = zip(*x)
    num_atoms = [len(v) for v in atom_vec]
    padding_final = np.max(num_atoms)
    padding_len = padding_final - num_atoms
    adj_mat = torch.stack(
        [F.pad(a, (0, p, 0, p), "constant", 0) for a, p in zip(adj_mat, padding_len)], 0
    )
    node_feat = torch.stack(
        [F.pad(n, (0, 0, 0, p), "constant", 0) for n, p in zip(node_feat, padding_len)], 0
    )
    atom_vec = torch.stack(
        [F.pad(v, (0, 0, 0, p), "constant", 0) for v, p in zip(atom_vec, padding_len)], 0
    )
    x = [adj_mat, node_feat, atom_vec]
    y = torch.stack(y, 0)
    return x, y


def get_multitask_loaders(source_path: str, batch_size: int) -> dict:
    """Get PyTorch DataLoader for multitask learning
    
    Parameters
    ----------
    source_path : str
        Path to data source directory
    batch_size : int
        Batch size
    
    Returns
    -------
    dict
        Dictionary of instantiated DataLoader for multitask learning
    """
    data = {}
    for file in ["featurized_data.pkl", "pretraining_split_idx.pkl", "meta_split_idx.pkl"]:
        try:
            with open(os.path.join(source_path, file), "rb") as f:
                data[file] = pickle.load(f)
        except FileNotFoundError as e:
            logging.error(f"File {file} not found in provided source_path {source_path}")
            sys.exit(1)

    loaders = {}
    meta_split_idx = data["meta_split_idx.pkl"]
    task_idx = list(set(meta_split_idx["meta_train"])) + list(set(meta_split_idx["meta_val"]))
    for split, idx in data["pretraining_split_idx.pkl"].items():
        x = [
            data["featurized_data.pkl"]["adj"][idx],
            data["featurized_data.pkl"]["feat"][idx],
        ]
        print(data["featurized_data.pkl"]["y"].shape)
        y = data["featurized_data.pkl"]["y"][idx][:, task_idx]
        loaders[split] = build_dataloader(
            x=x,
            y=y,
            batch_size=batch_size if split == "train" else 512,
            shuffle=True if split == "train" else False,
            num_workers=0,
        )

    return loaders
