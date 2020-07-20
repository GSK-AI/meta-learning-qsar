import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from src.models.ggnn import GatedGraphNeuralNetwork

PATH = Path(__file__).parent
OUTPUT_PATH = PATH / "output" / "maml"
SEEDS = ["0", "1", "2"]
EXPECTED_LOSSES = {
    "0": [0.4076, 0.0553, 0.0132, 0.0099, 0.0079, 0.0053],
    "1": [0.4559, 0.0398, 0.0080, 0.0024, 0.0017, 0.0016],
    "2": [0.7500, 0.1617, 0.0361, 0.0111, 0.0036, 0.0018],
}


@pytest.mark.dependency()
@pytest.mark.parametrize("seed", SEEDS)
def test_train_maml_integration(seed):
    """Test train_maml.py completes run"""
    cmd = []
    cmd.extend(["python", "src/train_maml.py"])
    cmd.extend(["--flagfile", f"{str(PATH/'configs/maml_flags.txt')}"])
    cmd.extend(["--source", f"{str(PATH/'data')}"])
    cmd.extend(["--save_path", str(OUTPUT_PATH / seed)])
    cmd.extend(["--seed", seed])
    output = subprocess.run(cmd)
    assert output.returncode == 0


@pytest.mark.dependency(depends=["test_train_maml_integration"])
@pytest.mark.parametrize("seed", SEEDS)
def test_train_maml_artifacts(seed):
    """Test train_maml.py producing expected artifacts"""
    assert "ckpts" in os.listdir(OUTPUT_PATH / seed)
    output_files = []
    output_files.append("flagfile.txt")
    output_files.append("summary.json")
    output_files.extend(
        [f"step_{s}{sd}.pth" for s in [1, 5, "init"] for sd in ["", "_state_dict"]]
    )
    for f in output_files:
        assert f in os.listdir(OUTPUT_PATH / f"{seed}/ckpts")


@pytest.mark.dependency(depends=["test_train_maml_artifacts"])
@pytest.mark.parametrize("seed", SEEDS)
def test_train_maml_loadable_checkpoints(seed):
    """Test train_maml.py produce loadable checkpoints"""
    checkpoints = [OUTPUT_PATH / f"{seed}/ckpts/step_{s}.pth" for s in [1, 5, "init"]]
    for f in checkpoints:
        model = torch.load(str(f))

    checkpoints_state_dict = [
        OUTPUT_PATH / f"{seed}/ckpts/step_{s}_state_dict.pth" for s in [1, 5, "init"]
    ]
    for f in checkpoints_state_dict:
        state_dict = torch.load(str(f))
        model = GatedGraphNeuralNetwork(n_edge=1, in_dim=75, n_conv=2, fc_dims=[32, 1])
        model.load_state_dict(state_dict)


@pytest.mark.dependency(depends=["test_train_maml_integration"])
@pytest.mark.parametrize("seed", SEEDS)
def test_train_maml_loss_trajectory(seed):
    """Test train_maml.py expected loss trajectory"""
    summary_dict = json.load(open(str(OUTPUT_PATH / f"{seed}/ckpts/summary.json"), "r"))
    losses = summary_dict["meta_training_loss"]
    expected_losses = EXPECTED_LOSSES[seed]
    assert losses == pytest.approx(expected_losses, abs=0.0001)
