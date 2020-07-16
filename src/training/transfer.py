from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import utils
from src.training.trainer import EarlyStopping, Trainer


def finetune_and_evaluate_pretrained_model(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    callbacks: list,
    dataloaders: Dict[str, DataLoader],
    epochs: int,
    metrics: List[str],
) -> Tuple[float, dict]:
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=device)

    best_model = trainer.train_dataloader(
        train_loader=dataloaders["train"],
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
        val_loader=dataloaders["val"],
    )

    results = evaluate_model(
        model=best_model,
        dataloader=dataloaders["test"],
        metrics=metrics,
        device=device,
        criterion=criterion,
    )

    return results


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    metrics: List[str],
    device: torch.device,
    criterion: Optional[torch.nn.Module] = None,
    task_index: Optional[int] = None,
    ignore_index: Optional[int] = None,
):
    model.to(device)
    if task_index is None:
        task_index = np.arange(dataloader.dataset.num_tasks)

    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for x, y_true in dataloader:
            x = utils.torch_utils.to_device(x, device)
            y_true = utils.torch_utils.to_device(y_true, device)
            y_pred = model(x)

            y_true_all.extend(y_true.cpu().data.numpy().tolist())
            y_pred_all.extend(y_pred.cpu().data.numpy().tolist())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    if task_index is not None:
        y_true_all = y_true_all[:, task_index]
        y_pred_all = y_pred_all[:, task_index]
    if ignore_index is not None:
        mask = y_true_all != ignore_index
        y_true_all = y_true_all[mask]
        y_pred_all = y_pred_all[mask]

    metrics_values = utils.metrics.calculate_metrics(
        metrics=metrics, y_true=y_true_all, y_pred=y_pred_all, threshold=0
    )

    results = {}
    results["metrics"] = metrics_values
    results["pred"] = y_pred_all.tolist()
    if criterion is not None:
        loss = criterion(torch.tensor(y_pred_all), torch.tensor(y_true_all)).item()
        results["loss"] = loss

    return results
