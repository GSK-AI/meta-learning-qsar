from typing import Dict, List

import numpy as np
import sklearn.metrics as skmetrics
import torch
from sklearn.neighbors import KNeighborsClassifier

from src import utils


def evaluate_knn_on_pretrained_model(
    model: torch.nn.Module,
    device: torch.device,
    loaders: Dict[str, torch.utils.data.DataLoader],
    metrics: List[str],
):
    model.eval()
    with torch.no_grad():
        x_train = []
        y_train = []
        for x, y_true in loaders["train"]:
            x = [x_.to(device) for x_ in x]
            x_train.extend(model.encode(x).cpu().data.numpy().tolist())
            y_train.extend(y_true.squeeze(-1).cpu().data.numpy().tolist())

        x_test = []
        y_true_all = []
        for x, y_true in loaders["test"]:
            x = [x_.to(device) for x_ in x]
            x_test.extend(model.encode(x).cpu().data.numpy().tolist())
            y_true_all.extend(y_true.squeeze(-1).cpu().data.numpy().tolist())

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_true_all = np.array(y_true_all)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    y_pred_all = neigh.predict_proba(x_test)[:, 1]

    metrics_values = utils.metrics.calculate_metrics(
        metrics=metrics, y_true=y_true_all, y_pred=y_pred_all, threshold=0.5
    )

    results = {}
    results["metrics"] = metrics_values
    results["pred"] = y_pred_all.tolist()
    return results
