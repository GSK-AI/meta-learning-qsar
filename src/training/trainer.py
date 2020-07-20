# Copyright Notice
# 
# Copyright (c) 2020 GlaxoSmithKline LLC (Kim Branson & Cuong Nguyen)
# Copyright (c) 2018 Victor Huang
# 
# This copyright work was created in 2020 and is based on the 
# copyright work created in 2018 available under the MIT License at 
# https://github.com/victoresque/pytorch-template

"""PyTorch trainer"""


import logging
import operator
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm.autonotebook import tqdm

from src.utils.logging import TqdmLoggingHandler, TqdmToLogger
from src.utils.torch_utils import to_device

log = logging.getLogger(__name__)
log.debug("Debugging")


class AverageBase(object):
    def __init__(self, value=0):
        self.value = float(value) if value is not None else None

    def __str__(self):
        return str(round(self.value, 4))

    def __repr__(self):
        return self.value

    def __format__(self, fmt):
        return self.value.__format__(fmt)

    def __float__(self):
        return self.value


class RunningAverage(AverageBase):
    """
    Keeps track of a cumulative moving average (CMA).
    """

    def __init__(self, value=0, count=0):
        super(RunningAverage, self).__init__(value)
        self.count = count

    def update(self, value):
        self.value = self.value * self.count + float(value)
        self.count += 1
        self.value /= self.count
        return self.value


class MovingAverage(AverageBase):
    """
    An exponentially decaying moving average (EMA).
    """

    def __init__(self, alpha=0.99):
        super(MovingAverage, self).__init__(None)
        self.alpha = alpha

    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * float(value)
        return self.value


def acc_logit(y_true, y_pred_logit):
    with torch.no_grad():
        y_pred = y_pred_logit.argmax(dim=1)
        correct = (y_pred == y_true).sum().float()
        accuracy_score = correct / len(y_pred)
    return accuracy_score.cpu().data.numpy()


class Trainer(object):
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        metrics=[],
        scheduler=None,
        clipnorm=None,
        device=torch.device("cpu"),
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.metrics = metrics
        self.train_losses = []
        self.val_losses = []
        self.scheduler = scheduler
        self.clipnorm = clipnorm
        self.best_loss_val = None
        self.best_metrics_val = None

    @property
    def summary(self):
        summary = {
            "history": {"loss": self.train_losses, "val_loss": self.val_losses},
            "best_loss_val": self.best_loss_val,
            "best_metrics_val": self.best_metrics_val,
        }
        return summary

    def _save_checkpoint(self, save_dir, epoch):
        m = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # state_dict only
        save_filename = f"epoch_{epoch+1}_state_dict.pth"
        save_path = os.path.join(save_dir, save_filename)
        torch.save(m.state_dict(), save_path)

        # everything
        save_filename = f"epoch_{epoch+1}.pth"
        save_path = os.path.join(save_dir, save_filename)
        state_dict = {
            "model": m,
            "model_state_dict": m.cpu().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": {"loss": self.train_losses, "val_loss": self.val_losses},
        }
        torch.save(state_dict, save_path)

        self.model.to(self.device)

    def train_dataloader(
        self,
        train_loader,
        epochs,
        save_dir=None,
        callbacks=[],
        val_loader=None,
        verbose=0,
    ):
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        if verbose == 2:
            tqdm_out = TqdmToLogger(log, level=logging.INFO)

        self.model = self.model.to(self.device)

        for epoch in range(epochs):
            start_time = time.time()

            epoch_message = f"Epoch {epoch+1}/{epochs}: "
            if verbose == 2:
                log.info(epoch_message)
                train_loader = tqdm(train_loader, file=tqdm_out)

            train_loss_epoch = self._train_single_epoch(train_loader, verbose)
            self.train_losses.append(train_loss_epoch)
            if val_loader is None:
                message = f"loss = {train_loss_epoch:.3f}\t"
                if verbose == 1:
                    message = epoch_message + message
                log.info(message)

            if val_loader is not None:
                val_loss_epoch, val_metrics_epoch = self._validate_single_epoch(
                    val_loader
                )
                self.val_losses.append(val_loss_epoch)
                if verbose > 0:
                    loss_message = f"loss = {train_loss_epoch:.3f}"
                    val_loss_message = f"val_loss = {val_loss_epoch:.3f}"
                    val_metrics_message = (
                        f"val_metrics = {[float('%.3f' % m) for m in val_metrics_epoch]}"
                        if isinstance(val_metrics_epoch, list)
                        else f"val_metrics = {val_metrics_epoch:.3f}"
                    )
                    message = f"{loss_message:<15}{val_loss_message:<20}{val_metrics_message:<20}"
                    if verbose == 1:
                        message = f"{epoch_message:<15}{message}"
                    log.info(message)

                if self.scheduler is not None:
                    self.scheduler.step(val_loss_epoch)

                cbs = [cb.step(val_loss_epoch, verbose) for cb in callbacks]
                if "best" in cbs:
                    best_model = deepcopy(self.model)
                    self.best_loss_val = val_loss_epoch
                    self.best_metrics_val = val_metrics_epoch
                    if save_dir is not None:
                        self._save_checkpoint(save_dir, epoch)

                if "break" in cbs:
                    break

            if verbose == 2:
                log.info(f"Elapsed time: {(time.time() - start_time):.3f} seconds")

            torch.cuda.empty_cache()

        if val_loader is None or len(callbacks) == 0:
            best_model = self.model

        return best_model

    def _train_single_epoch(self, train_loader, verbose):
        self.model.train()
        train_loss = MovingAverage()

        for x, y_true in train_loader:
            # model's forward pass on correct device
            x = to_device(x, self.device)
            y_true = to_device(y_true, self.device)
            y_pred = self.model(x)
            # calculate the loss
            if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                loss = self.criterion(y_pred, torch.squeeze(y_true, dim=1).long())
            else:
                loss = self.criterion(y_pred, y_true)
            # clear previous gradient computation and calculate gradients
            self.optimizer.zero_grad()
            loss.backward()

            # update model weights
            if self.clipnorm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clipnorm)
            self.optimizer.step()
            train_loss.update(loss.item())

            # del x, y_true, y_pred, loss
            # torch.cuda.empty_cache()

        return train_loss.value

    @staticmethod
    def _calculate_single_metric(metric, y_true, y_pred, *args):
        args = list(args)
        for i, a in enumerate(args):
            if isinstance(a, list):
                args[i] = [a_.cpu().data.numpy() for a_ in a]
            else:
                args[i] = a.cpu().data.numpy()

        if metric == "acc":
            metric_result = acc_logit(y_true.long(), y_pred)
        else:
            metric_result = metric(
                y_true.cpu().data.numpy(), y_pred.cpu().data.numpy(), *args
            )
        return metric_result

    def _validate_single_epoch(self, val_loader):
        self.model.eval()
        valid_loss = RunningAverage()

        # check for specific metrics per output
        if len(self.metrics) == 0:
            metrics = self.metrics
        elif isinstance(self.metrics[0], list):
            metrics = [[np.empty(0) for _ in range(len(m))] for m in self.metrics]
        else:
            metrics = [np.empty(0) for _ in range(len(self.metrics))]
        # keep track of predictions
        y_pred = []

        with torch.no_grad():

            for x, y_true in val_loader:
                # model's forward pass on correct device
                x = to_device(x, self.device)
                y_true = to_device(y_true, self.device)
                y_pred = self.model(x)  # calculate the loss
                if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                    loss = self.criterion(y_pred, torch.squeeze(y_true, dim=1).long())
                else:
                    loss = self.criterion(y_pred, y_true)

                # TODO: make this recursive
                # update metrics
                for i, metric in enumerate(self.metrics):
                    if isinstance(metric, list):
                        for j, m in enumerate(metric):
                            metric_value = self._calculate_single_metric(
                                m, y_true[:, i], y_pred[:, i]
                            )
                            metrics[i][j] = np.append(metrics[i][j], metric_value)

                    else:
                        metric_value = self._calculate_single_metric(
                            metric, y_true, y_pred, x
                        )
                        metrics[i] = np.append(metrics[i], metric_value)
                # update running loss value
                valid_loss.update(loss.item())

                del x, y_true, y_pred, loss
                torch.cuda.empty_cache()

        if len(self.metrics) > 0:
            metrics = [np.mean(m) for m in metrics]
        else:
            metrics = valid_loss.value

        return valid_loss.value, metrics


class EarlyStopping(object):
    def __init__(self, patience, mode="min"):
        assert mode in ["min", "max"], "mode must be one of ['min', 'max']"
        self.max_patience = patience
        self.current_patience = 0
        self.best_loss = None
        self.op = self._set_mode(mode)

    @staticmethod
    def _set_mode(mode):
        if mode == "min":
            op = operator.gt
        elif mode == "max":
            op = operator.lt
        return op

    def step(self, loss, verbose):
        if self.best_loss is None:
            self.best_loss = loss
        if self.op(loss, self.best_loss):
            self.current_patience += 1
            if self.current_patience == self.max_patience:
                log.info("Early stopping yooo!")
                return "break"
            else:
                return "pass"
        else:
            if verbose > 0:
                log.info(
                    f"Early stopping metric improves from {self.best_loss:.3f} to {loss:.3f}"
                )
            self.current_patience = 0
            self.best_loss = loss
            return "best"
