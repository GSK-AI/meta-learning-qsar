import pytest
import logging
import numpy as np
from openeye import oechem
from src.featurizer.molgraph import Adjacency, Nodes, MolGraph


TEST_SMILES = ["CCC", "c1ccccc1"]
TEST_ADJ = [
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    [
        [0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
    ],
]


@pytest.mark.parametrize("smiles, expected_adj", list(zip(TEST_SMILES, TEST_ADJ)))
def test_adjacency(smiles, expected_adj):
    if not oechem.OEChemIsLicensed():
        logging.warning(
            "License for OpenEye OEChem TK is not found. Not testing featurizers."
        )
        return True

    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)
    oechem.OESuppressHydrogens(mol)
    adj = Adjacency(mol=mol)
    assert adj.adj_mat.tolist() == expected_adj


def test_normalization():
    TEST_MAT = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    TEST_NORM = TEST_MAT / 4
    norm = Adjacency._normalize_adj_mat(TEST_MAT)
    assert norm.tolist() == TEST_NORM.tolist()


@pytest.mark.parametrize(
    "x, allowable_set, unk, expected_encoding",
    [
        (1, [0, 1, 2], False, [0, 1, 0]),
        ("b", ["a", "b", "c"], False, [0, 1, 0]),
        (2, [0, 1, "unk"], True, [0, 0, 1]),
    ],
)
def test_encoding(x, allowable_set, unk, expected_encoding):
    encoding = Nodes.one_of_k_encoding(x=x, allowable_set=allowable_set, unk=unk)
    assert encoding == expected_encoding


def test_encoding_exception():
    with pytest.raises(Exception):
        encoding = Nodes.one_of_k_encoding(x=1, allowable_set=[], unk=False)
