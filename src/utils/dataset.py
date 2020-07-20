"""PyTorch Dataset for graphs"""

from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, x: List[List[np.ndarray]], y: Optional[np.ndarray] = None):
        """PyTorch Dataset for molecular graphs.

        Parameters
        ----------
        x : List[List[np.ndarray]]
            List of adjacency matrices and node feature matrices.
            Must be in the following order [adj_mat, node_feat] where
            each of adj_mat and node_feat is a list of np.ndarray.
        y : np.ndarray, optional
            Labels, by default None
        """
        super(GraphDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x[0])

    @property
    def num_tasks(self):
        if len(self.y.shape) == 1:
            return 1
        else:
            return self.y.shape[-1]

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Get item from dataset by indexing.

        Parameters
        ----------
        idx : int
            Item index in dataset

        Returns
        -------
        Tuple[List[torch.Tensor], torch.Tensor]
            Tuple of 2 elements (x, y).
            x is a list of [adj_mat, node_feat, atom_vec] with shapes:
                - adj_mat: (N, N)
                - node_feat: (N, F)
                - atom_vec: (N, 1)
            where N is number of nodes and F is number of features.
            y is the output labels. y is set to 0 if no outputs are provided.
        """
        x = [torch.FloatTensor(self.x[0][idx]), torch.FloatTensor(self.x[1][idx])]
        a = torch.ones(len(x[1]), 1)
        x.append(a)

        try:
            y = torch.tensor(self.y[idx])
        except TypeError:
            y = torch.tensor(0)
        finally:
            if len(y.size()) < 1:
                y = y.unsqueeze(-1)
        return x, y
