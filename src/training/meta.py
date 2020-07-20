import json
import logging
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src import utils


def fast_adapt(
    adaptation_data: Tuple[torch.Tensor],
    evaluation_data: Tuple[torch.Tensor],
    learner,
    criterion: nn.Module,
    inner_steps: int,
    device: torch.device,
    metrics: List[str],
    no_grad: bool = False,
):
    """Fast adaptation for inner loop training"""
    for step in range(inner_steps):
        x, y = adaptation_data
        x = utils.torch_utils.to_device(x, device)
        y = utils.torch_utils.to_device(y, device)
        y_pred = learner(x)
        train_loss = criterion(y_pred, y)
        train_loss /= len(y)
        learner.adapt(train_loss)

    x, y = evaluation_data
    x = utils.torch_utils.to_device(x, device)
    y = utils.torch_utils.to_device(y, device)
    if no_grad:
        with torch.no_grad():
            y_pred = learner(x)
            valid_loss = criterion(y_pred, y)
    else:
        y_pred = learner(x)
        valid_loss = criterion(y_pred, y)

    valid_metrics = utils.metrics.calculate_metrics(
        metrics=metrics,
        y_true=y.cpu().data.numpy(),
        y_pred=y_pred.cpu().data.numpy(),
        threshold=0,
    )

    del x, y
    torch.cuda.empty_cache()

    return valid_loss, valid_metrics


def meta_training(
    meta_learner,
    meta_steps: int,
    meta_batch_size: int,
    loaders: dict,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    inner_steps: int,
    device: torch.device,
    save_path: str,
    ckpt_steps: int,
    metrics: List[str] = ["accuracy_score"],
):
    if meta_batch_size > len(loaders["meta_train"]["train"]):
        raise ValueError(
            f"meta_batch_size ({meta_batch_size}) must <= number of tasks ({len(loaders['meta_train']['train'])})"
        )
    summary_dict = {}
    summary_dict["meta_training_loss"] = []
    summary_dict["meta_training_metrics"] = {m: [] for m in metrics}

    save_checkpoint(model=meta_learner.module, save_path=save_path, step="init")
    logging.info("Initial checkpoint saved.")

    for step in range(meta_steps):
        logging.info(f"iteration {step}")

        iteration_loss = 0.0
        meta_train_loss = 0.0
        meta_training_metrics = {m: 0 for m in metrics}

        random_task_indices = np.random.choice(
            len(loaders["meta_train"]["train"]),
            size=meta_batch_size,
            replace=False,
        )
        for task_index in random_task_indices:
            # Compute meta-training loss
            learner = meta_learner.clone()
            task_dataloader = iter(loaders["meta_train"]["train"][task_index])
            inner_data = task_dataloader.next()
            outer_data = task_dataloader.next()
            outer_loss, outer_metrics = fast_adapt(
                adaptation_data=inner_data,
                evaluation_data=outer_data,
                learner=learner,
                criterion=criterion,
                inner_steps=inner_steps,
                device=device,
                metrics=metrics,
            )
            iteration_loss += outer_loss
            meta_train_loss += outer_loss.item()
            meta_training_metrics = {
                k: v + outer_metrics[k]
                for k, v in meta_training_metrics.items()
            }

        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()

        meta_train_loss /= meta_batch_size
        summary_dict["meta_training_loss"].append(meta_train_loss)
        logging.info(f"meta_training_loss: {meta_train_loss}")

        # Print some metrics
        for k, v in meta_training_metrics.items():
            v /= meta_batch_size
            summary_dict["meta_training_metrics"][k].append(v)
            logging.info(f"{k}: {v}")

        with open(os.path.join(save_path, "summary.json"), "w") as f:
            json.dump(summary_dict, f)

        if (step + 1) % ckpt_steps == 0 or step == 0:
            save_checkpoint(
                model=meta_learner.module, save_path=save_path, step=step + 1
            )
            logging.info("Checkpoint saved.")

        torch.cuda.empty_cache()


def save_checkpoint(model: nn.Module, save_path: str, step: Union[str, int]):
    ckpt_name = f"step_{step}"
    torch.save(model, os.path.join(save_path, f"{ckpt_name}.pth"))
    torch.save(
        model.state_dict(),
        os.path.join(save_path, f"{ckpt_name}_state_dict.pth"),
    )
