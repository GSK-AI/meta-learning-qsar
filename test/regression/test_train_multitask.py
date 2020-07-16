import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from src.models.ggnn import GatedGraphNeuralNetwork

PATH = Path(__file__).parent
OUTPUT_PATH = PATH / "output" / "multitask"
SEEDS = ["0", "1", "2"]
EXPECTED_LOSSES = {
    "0": {"loss": [0.7006, 0.6902], "val_loss": [0.6885, 0.6786]},
    "1": {"loss": [0.6975, 0.6880], "val_loss": [0.6864, 0.6777]},
    "2": {"loss": [0.6954, 0.6845], "val_loss": [0.6829, 0.6718]},
}


@pytest.mark.dependency()
@pytest.mark.parametrize("seed", SEEDS)
def test_train_multitask_integration(seed):
    """Test train_multitask.py completes run"""
    cmd = []
    cmd.extend(["python", "src/train_multitask.py"])
    cmd.extend(["--flagfile", f"{str(PATH/'configs/multitask_flags.txt')}"])
    cmd.extend(["--source", f"{str(PATH/'data')}"])
    cmd.extend(["--save_path", str(OUTPUT_PATH / seed)])
    cmd.extend(["--seed", seed])
    output = subprocess.run(cmd)
    assert output.returncode == 0


@pytest.mark.dependency(depends=["test_train_multitask_integration"])
@pytest.mark.parametrize("seed", SEEDS)
def test_train_multitask_artifacts(seed):
    """Test train_multitask.py producing expected artifacts"""
    output_files = []
    output_files.append("flagfile.txt")
    output_files.append("summary.json")
    output_files.append("best_model.pth")
    output_files.extend(
        [f"epoch_{s}{sd}.pth" for s in [1, 2] for sd in ["", "_state_dict"]]
    )
    for f in output_files:
        assert f in os.listdir(OUTPUT_PATH / seed)


@pytest.mark.dependency(depends=["test_train_multitask_artifacts"])
@pytest.mark.parametrize("seed", SEEDS)
def test_train_multitask_loadable_checkpoints(seed):
    """Test train_multitask.py produce loadable checkpoints"""
    checkpoints = [OUTPUT_PATH / f"{seed}/epoch_{s}.pth" for s in [1, 2]]
    for f in checkpoints:
        model = torch.load(str(f))

    checkpoints_state_dict = [
        OUTPUT_PATH / f"{seed}/epoch_{s}_state_dict.pth" for s in [1, 2]
    ]
    for f in checkpoints_state_dict:
        state_dict = torch.load(str(f))
        model = GatedGraphNeuralNetwork(n_edge=1, in_dim=75, n_conv=2, fc_dims=[32, 879])
        model.load_state_dict(state_dict)


@pytest.mark.dependency(depends=["test_train_multitask_integration"])
@pytest.mark.parametrize("seed", SEEDS)
def test_train_multitask_loss_trajectory(seed):
    """Test train_multitask.py expected loss trajectory"""
    summary_dict = json.load(open(str(OUTPUT_PATH / f"{seed}/summary.json"), "r"))
    losses = summary_dict["history"]
    expected_losses = EXPECTED_LOSSES[seed]
    for k in losses.keys():
        assert losses[k] == pytest.approx(expected_losses[k], abs=0.0001)
