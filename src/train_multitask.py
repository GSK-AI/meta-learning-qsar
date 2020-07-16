#!/usr/bin/env python3
import copy
import gc
import json
import os
import pickle
import random
from functools import partial

import learn2learn as l2l
import numpy as np
import torch
from absl import app, flags, logging
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.models.ggnn import GatedGraphNeuralNetwork
from src.training.trainer import EarlyStopping, Trainer
from src.utils import dataloaders, loss, torch_utils

FLAGS = flags.FLAGS
flags.DEFINE_string("save_path", None, "Misc: Folder directory for saving MAML models")
flags.DEFINE_string("source", None, "Training: Data folder")
flags.DEFINE_integer("seed", 0, "Training: Random seed")
flags.DEFINE_integer("n_conv", 7, "Architecture: Number of gated graph convolution layers")
flags.DEFINE_integer("fc_dims", 1024, "Architecture: Number of fully connected layers")
flags.DEFINE_float("lr", -3.5, "Training: Learning rate")
flags.DEFINE_integer("epochs", 200, "Training: Number of epochs")
flags.DEFINE_integer("batch_size", 64, "Training: batch size")
flags.mark_flag_as_required("save_path")
flags.mark_flag_as_required("source")


def train_multitask(argv):
    logging.info(f"Starting multitask training with {FLAGS.source} datset.")
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)
    get_path = partial(os.path.join, FLAGS.save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FLAGS.flags_into_string()
    FLAGS.append_flags_into_file(get_path("flagfile.txt"))

    logging.info(f"Setting seed...")
    torch_utils.set_seed(FLAGS.seed)

    logging.info(f"Loading data...")
    loaders = dataloaders.get_multitask_loaders(
        source_path=FLAGS.source, batch_size=FLAGS.batch_size
    )
    model = GatedGraphNeuralNetwork(
        n_edge=1,
        in_dim=75,
        n_conv=FLAGS.n_conv,
        fc_dims=[FLAGS.fc_dims, loaders["train"].dataset.num_tasks],
    )
    model = model.to(device)

    criterion = loss.MaskedBCEWithLogitsLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=10 ** (FLAGS.lr))
    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer, device=device
    )

    cbs = [EarlyStopping(patience=20, mode="min")]

    logging.info(f"Begin training!")
    best_model = trainer.train_dataloader(
        train_loader=loaders["train"],
        epochs=FLAGS.epochs,
        callbacks=cbs,
        verbose=1,
        save_dir=FLAGS.save_path,
        val_loader=loaders["val"],
    )
    summary = trainer.summary
    torch.save(best_model, get_path("best_model.pth"))
    json.dump(summary, open(get_path("summary.json"), "w"))
    logging.info(f"""Summary is saved at {get_path("summary.json")}""")


if __name__ == "__main__":
    app.run(train_multitask)
