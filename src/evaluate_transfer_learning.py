"""Evaluate transfer learning

"""

import datetime
import json
import logging
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import learn2learn as l2l
import numpy as np
import torch
from absl import app, flags, logging
from sklearn.model_selection import train_test_split
from torch import nn, optim

from src.models.ggnn import GatedGraphNeuralNetwork
from src.training.knn import evaluate_knn_on_pretrained_model
from src.training.meta import meta_training
from src.training.trainer import EarlyStopping
from src.training.transfer import finetune_and_evaluate_pretrained_model
from src.utils import dataloaders, torch_utils

FLAGS = flags.FLAGS
flags.DEFINE_string("init_path", None, "Path to PyTorch initialization (this should be a file output by torch.save)")
flags.DEFINE_string("source", None, "Data folder (see train_maml.py documentation for requirements)")
flags.DEFINE_boolean("multitask", False, "Model specified in --init_path is multitask baseline (setting this flag will reinitialize output layer)")
flags.DEFINE_string("mode", "binary_classification", "Task mode in [regression, binary_classification] (This will set the loss function)")
flags.DEFINE_boolean("freeze", False, "Freeze parameters up to penultimate layer (used with --multitask)")
flags.DEFINE_integer("num_data", 0, "Number of datapoints to finetune on")
flags.DEFINE_float("lr", 0.005, "Learning rate for finetuning")
flags.DEFINE_bool("test_set", False, "Evaluate on test set instead of validation set")
flags.DEFINE_bool("knn", False, "Use kNN on the penultimate layer of the model provided in --init_path")
flags.DEFINE_bool("anil", False, "Model provided in --init_path is trained by ANIL")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_integer("split_seed", 0, "scikit-learn random seed")
flags.DEFINE_list("metrics", "accuracy_score,average_precision_score", "Metrics")
flags.DEFINE_string("filename_ext", "", "Extensions added to filename for saving")
flags.mark_flag_as_required("init_path")
flags.mark_flag_as_required("source")


def eval_transfer(argv):
    """Evaluate transfer learning from pretrained model"""
    torch_utils.set_seed(FLAGS.seed)
    loaders = dataloaders.get_loaders(source_path=FLAGS.source)

    if FLAGS.test_set:
        loaders = loaders["meta_test"]
    else:
        loaders = loaders["meta_val"]
        loaders["test"] = loaders["val"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = []
    losses = []
    metrics = {m: [] for m in FLAGS.metrics}
    for l_train, l_val, l_test in zip(
        loaders["train"], loaders["val"], loaders["test"]
    ):
        torch_utils.set_seed(FLAGS.seed)
        size = len(l_train.dataset) + len(l_val.dataset) + len(l_test.dataset)
        stratify = None if FLAGS.mode == "regression" else l_train.dataset.y
        if FLAGS.num_data not in [None, 0]:
            adj_adapt, _, feat_adapt, _, y_adapt, _ = train_test_split(
                *l_train.dataset.x,
                l_train.dataset.y,
                train_size=FLAGS.num_data,
                stratify=stratify,
                random_state=FLAGS.split_seed,
            )
            batch_size = l_train.batch_size
            batch_size = batch_size if batch_size > FLAGS.num_data else FLAGS.num_data

            l_train = dataloaders.build_dataloader(
                x=[adj_adapt, feat_adapt],
                y=y_adapt,
                batch_size=batch_size,#min([l_train.batch_size, FLAGS.num_data]),
                shuffle=True,
                num_workers=0,
            )

        logging.info(f"Loading initializations from {FLAGS.init_path}")
        model = torch.load(FLAGS.init_path)

        if FLAGS.multitask:
            model.transfer(out_dim=1, freeze=FLAGS.freeze)

        if FLAGS.anil:
            params = list(model.parameters())
            for i, p in enumerate(params):
                if i < (len(params) - 2):
                    p.requires_grad = False

        setattr(model, "dropout", nn.Dropout(0.0))

        model = model.to(device)

        if FLAGS.knn:
            results = evaluate_knn_on_pretrained_model(
                model=model,
                device=device,
                loaders={"train": l_train, "test": l_test},
                metrics=FLAGS.metrics,
            )
            results["loss"] = 0
        else:
            if FLAGS.mode == "binary_classification":
                if FLAGS.test_set:
                    pos_weight = (l_train.dataset.y == 0).sum() / l_train.dataset.y.sum()
                    # pos_weight = max([pos_weight, -5])
                    # pos_weight = min([pos_weight, 5])
                    pos_weight = torch.tensor(pos_weight)
                else:
                    pos_weight = None
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            elif FLAGS.mode == "regression":
                criterion = nn.MSELoss()
                
            optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
            cbs = [EarlyStopping(patience=10, mode="min")]
            epochs = 200
            task_dataloaders = {"train": l_train, "val": l_val, "test": l_test}

            results = finetune_and_evaluate_pretrained_model(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                callbacks=cbs,
                dataloaders=task_dataloaders,
                epochs=epochs,
                metrics=FLAGS.metrics,
            )

        predictions.append(results["pred"])
        losses.append(results["loss"])
        logging.info(f"loss: {losses[-1]}")
        for k, v in results["metrics"].items():
            metrics[k].append(v)
            logging.info(f"{k}: {v}")
        logging.info(f"size: {size}")
        
    # Write summaries
    init_path = FLAGS.init_path.split("/")
    save_path = "/".join(init_path[:-2])
    save_path = os.path.join(save_path, "eval_summary")
    os.makedirs(save_path, exist_ok=True)

    write_summary(
        loss=losses,
        metrics=metrics,
        predictions=predictions,
        save_path=save_path,
        model_name=init_path[-1],
        flags=FLAGS,
    )


def write_summary(
    loss: List[float], metrics: dict, predictions: list, save_path: str, model_name: str, flags: Any
):
    """Write summary of evaluation"""
    mean_metrics = {f"mean_{k}": np.mean(v).tolist() for k, v in metrics.items()}
    metrics.update(mean_metrics)
    for k, v in mean_metrics.items():
        logging.info(f"{k}: {v}")

    timestamp = str(datetime.datetime.now())
    metadata = [f.serialize() for f in FLAGS.get_key_flags_for_module(sys.argv[0])]

    summary = {}
    summary["metadata"] = metadata
    summary["timestamp"] = timestamp
    summary["loss"] = loss
    summary["metrics"] = metrics
    summary["pred"] = predictions
    source = FLAGS.source.split("/")[-1]
    summary_filename = [model_name, str(FLAGS.num_data), source, str(FLAGS.split_seed)]
    if FLAGS.multitask:
        summary_filename.append("multitask")
        if FLAGS.freeze:
            summary_filename.append("freeze")
        if FLAGS.knn:
            summary_filename.append("knn")
    if FLAGS.test_set:
        summary_filename.append("test")
    if FLAGS.filename_ext:
        summary_filename.append(FLAGS.filename_ext)
    
    summary_filename = "_".join(summary_filename) + ".json"
    summary_path = os.path.join(save_path, summary_filename)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary statistics are saved at {summary_path}")


if __name__ == "__main__":
    app.run(eval_transfer)
