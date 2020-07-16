"""
Dealing with different devices etc..
"""

from typing import Union

import numpy as np
import torch


def to_device(
    x: Union[torch.Tensor, np.ndarray, dict], device: str
) -> Union[torch.Tensor, dict]:
    """
    Sends x to the specified device.
    If x is a numpy array, it gets cast to a pytorch tensor on the device.
    If x is a dictionary, the function is applied recursively to the values in the dictionary.
    Parameters
    ----------
    x : Union[torch.Tensor, np.ndarray, dict]
        object to send to device
    device : str
        specifies device
    Returns
    -------
    Union[torch.Tensor, dict]
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        return torch.tensor(x, device=device)
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = to_device(v, device=device)
        return x
    if isinstance(x, (list, tuple)):
        x = [to_device(x_, device=device) for x_ in x]
        return x


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
