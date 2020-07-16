"""Custom masked binary cross entropy loss"""

from typing import List, Set, Dict, Tuple, Optional, Iterable

import torch
import torch.nn as nn
import sys


class MaskedBCEWithLogitsLoss(nn.Module):
    """Masked BCE with logits loss
    
    Use this class in place of torch.nn.BCEWithLogitsLoss to ignore missing values
    as indicated by ignore_index.
    """

    def __init__(self, ignore_index=-1, **kwargs):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction="none", **kwargs)
        self.ignore_index = ignore_index
        self.register_buffer("ignore_class", torch.Tensor([float(0)]))
        self.register_buffer("ignore_logit", torch.Tensor([float("-10e8")]))

    def forward(self, output, target):
        mask = target != self.ignore_index
        loss = self.criterion(output, target)
        loss = torch.masked_select(loss, mask).sum() / mask.sum()
        return loss

