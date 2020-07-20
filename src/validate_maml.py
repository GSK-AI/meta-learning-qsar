"""Asynchronous validation for MAML training.

The script monitors <monitor_path>/<ckpts_folder> and kick-off a validation step using slurm 
when new checkpoints are found. It saves a JSON with validation metrics to <monitor_path>/<summary_folder>.
The script automatically stops after 1 hour of no new checkpoints.




Usage (with src/train_maml.py running):
    python src/validate_maml.py \
        --monitor_path <directory to store checkpoint> \
        --source <directory where training and validation data is stored> \
        ...

Notes:
    - --monitor_path here --save_path in src/train_maml.py must be the same
    - --source here and in src/train_maml.py must be the same
"""
import json
import os
import subprocess
import time
from functools import partial
from typing import Tuple

import numpy as np
from absl import app, flags, logging

from src.utils import slurm_utils

FLAGS = flags.FLAGS
# Training directory and subfolder names for monitoring
flags.DEFINE_string("monitor_path", None, "Path start monitoring for MAML checkpoints")
flags.DEFINE_string("ckpts_folder", "ckpts", "Checkpoints folder name in <monitor_path>")
flags.DEFINE_string("summary_folder", "eval_summary", "Summary folder name in <monitor_path>")
# Validation settings
flags.DEFINE_string("mode", "binary_classification", "Validation: [regression, binary_classification]")
flags.DEFINE_string("seed", "0", "Validation: Number of meta gradient steps to take")
flags.DEFINE_string("source", None, "Validation: Data folder (see docstrings for requirements)")
flags.DEFINE_integer("num_data", 0, "Validation: Number of instances to finetune on")
flags.DEFINE_bool("anil", False, "Validation: Use the ANIL algorithm from DeepMind")
flags.DEFINE_bool("test_set", False, "Validation: Evaluate on test set instead of validation set")
flags.DEFINE_boolean("multitask", False, "Validation: Pretrained model is multitask baseline")
flags.DEFINE_string("metrics", "average_precision_score,accuracy_score", "Validation: Metrics")
# Slurm resources
flags.DEFINE_string("conda_env", "metalearning", "Conda environment for validation slurm jobs")
flags.DEFINE_string("partition", "aiml", "Partition for running validation slurm jobs")
flags.DEFINE_string("gres", "gpu:1", "Generic resources for slurm jobs")
flags.DEFINE_string("mem", "32G", "CPU memory for slurm jobs")

flags.mark_flag_as_required("monitor_path")
flags.mark_flag_as_required("source")


def validate_maml(argv):
    ckpts_path = os.path.join(FLAGS.monitor_path, FLAGS.ckpts_folder)
    summary_path = os.path.join(FLAGS.monitor_path, FLAGS.summary_folder)

    os.makedirs(ckpts_path, exist_ok=True)
    os.makedirs(summary_path, exist_ok=True)

    ckpts = os.listdir(ckpts_path)
    ckpts = [f for f in ckpts if ".pth" in f and "state_dict" not in f]
    eval_summary = os.listdir(summary_path)
    eval_summary = [f for f in eval_summary if ".json" in f]

    ckpts_to_test = [
        f for f in ckpts if not any([f.split(".")[0] in s for s in eval_summary])
    ]
    ckpts_done = [f for f in ckpts if f not in ckpts_to_test]

    iterations_with_no_new_ckpt = 0
    while True:
        logging.info(f"New checkpoints: {ckpts_to_test}")
        logging.info(f"Completed checkpoints: {ckpts_done}")
        for ckpt in ckpts_to_test:
            logging.info(f"Testing {ckpt}")

            ckpt_path = os.path.join(ckpts_path, ckpt)
            log_path = os.path.join(FLAGS.monitor_path, "logs")
            os.makedirs(log_path, exist_ok=True)
            log_path = os.path.join(log_path, f"{ckpt}.log")

            wrap_cmd = []
            wrap_cmd.append(f"python src/evaluate_transfer_learning.py")
            wrap_cmd.append(f"--init_path {ckpt_path}")
            wrap_cmd.append(f"--source {FLAGS.source}")
            wrap_cmd.append(f"--mode {FLAGS.mode}")
            wrap_cmd.append(f"--seed {FLAGS.seed}")
            wrap_cmd.append(f"--num_data {FLAGS.num_data}")
            wrap_cmd.append(f"--metrics {FLAGS.metrics}")
            if FLAGS.anil:
                wrap_cmd.append("--anil")
            if FLAGS.multitask:
                wrap_cmd.append("--multitask")
            if FLAGS.test_set:
                wrap_cmd.append("--test_set")
            wrap_cmd = " ".join(wrap_cmd)

            job_id = slurm_utils.batch_job(
                command=wrap_cmd,
                modules=["anaconda3"],
                conda_env=FLAGS.conda_env,
                cores=4,
                partition=FLAGS.partition,
                gres=FLAGS.gres,
                mem=FLAGS.mem,
                requested_time="5-00",
            )

            logging.info(wrap_cmd)
            logging.info(f"Submmitted Job {job_id}")
            ckpts_done.append(ckpt)

        best_step_ckpt_file, best_step_eval_summary_file = save_best_model(
            summary_path=summary_path,
            ckpts_path=ckpts_path,
            metric=f"mean_{FLAGS.metrics.split(',')[0]}",
        )
        logging.info("Submitted validation for all new checkpoints.")
        logging.info(f"Best model: {best_step_ckpt_file}")
        logging.info(f"Validation: {best_step_eval_summary_file}")
        logging.info("Sleep for 5 minutes.")

        time.sleep(300)

        ckpts = os.listdir(ckpts_path)
        # drop state dict files
        ckpts = [f for f in ckpts if ".pth" in f and "state_dict" not in f]
        # drop checkpoints already completed
        ckpts = [f for f in ckpts if f not in ckpts_done]
        # drop checkpoints that are symlinked
        ckpts_to_test = [f for f in ckpts if not os.path.islink(os.path.join(ckpts_path, f))]

        is_empty = not bool(ckpts_to_test)
        iterations_with_no_new_ckpt += int(is_empty)
        iterations_with_no_new_ckpt *= int(is_empty)
        logging.info(f"{is_empty}, {iterations_with_no_new_ckpt}")
        if iterations_with_no_new_ckpt >= 12:
            logging.info("No new checkpoint found in 60 minutes. Exiting.")
            break


def save_best_model(summary_path: str, ckpts_path: str, metric: str) -> Tuple[str]:
    """Check validation performance and return checkpoint and summary filename for best model"""
    # Get paths to summary files, drop best_model symlinked file and any file with "test"
    eval_summary_paths = [os.path.join(summary_path, f) for f in os.listdir(summary_path) if "test" not in f]
    eval_summary_paths = [f for f in eval_summary_paths if not os.path.islink(f)] 
    if len(eval_summary_paths) == 0:
        return None, None

    # Get metrics of interest
    metrics = [json.load(open(f, "r"))["metrics"] for f in eval_summary_paths]
    metrics = np.array([m[metric] for m in metrics])
    if len(metrics.shape) > 1:
        metrics = np.mean(metrics, axis=1)

    # Get best model step
    idx = metrics.argsort()[-3:][::-1][0]
    eval_summary_files = [os.path.basename(f) for f in eval_summary_paths]
    best_step_eval_summary_file = eval_summary_files[idx]
    best_step = best_step_eval_summary_file.split("_")[1]
    best_step_ckpt_file = [f for f in os.listdir(ckpts_path) if f"_{best_step}" in f]
    best_step_ckpt_file = best_step_ckpt_file[0]
    
    # symlink for best checkpoint to best_model.pth
    best_ckpt_filename = os.path.join(ckpts_path, "best_model.pth")
    best_ckpt_filename = os.path.abspath(best_ckpt_filename)
    best_step_ckpt_file = os.path.join(ckpts_path, best_step_ckpt_file)
    best_step_ckpt_file = os.path.abspath(best_step_ckpt_file)
    if os.path.exists(best_ckpt_filename) and os.path.islink(best_ckpt_filename):
        os.unlink(best_ckpt_filename)
    os.symlink(best_step_ckpt_file, best_ckpt_filename)

    # and symlink for best summary to best_model_eval_summary.json
    best_summary_filename = os.path.join(summary_path, "best_model_eval_summary.json")
    best_summary_filename = os.path.abspath(best_summary_filename)
    best_step_eval_summary_file = os.path.join(summary_path, best_step_eval_summary_file)
    best_step_eval_summary_file = os.path.abspath(best_step_eval_summary_file)
    if os.path.exists(best_summary_filename) and os.path.islink(best_summary_filename):
        os.unlink(best_summary_filename)
    os.symlink(best_step_eval_summary_file, best_summary_filename)

    return (best_step_ckpt_file, best_step_eval_summary_file)


if __name__ == "__main__":
    app.run(validate_maml)
