# Copyright Notice
# 
# Copyright (c) 2020 GlaxoSmithKline LLC (Kim Branson & Cuong Nguyen)
# Copyright (c) 2017 PandeLab
# 
# This copyright work was created in 2020 and is based on the 
# copyright work created in 2017 available under the MIT License at 
# https://github.com/deepchem/deepchem/blob/master/deepchem/feat/graph_features.py

"""Molecule graph object that converts SMILES strings to graphs using OpenEye OEChem Toolkit"""

import numpy as np
from openeye import oechem
from typing import Union, Any, Optional


class Adjacency:
    """Adjacency class for MolGraph"""

    def __init__(self, mol: oechem.OEMol):
        """
        mol (oechem.OEMol): OEMol object
        """
        self.num_atoms = mol.NumAtoms()
        self.adj_mat = self._get_adjacency_matrix(mol)

    def _get_adjacency_matrix(self, mol:oechem.OEMol) -> np.ndarray:
        adj_mat = np.zeros((self.num_atoms, self.num_atoms))
        for bond in mol.GetBonds():
            bgn_idx = bond.GetBgnIdx()
            end_idx = bond.GetEndIdx()
            adj_mat[bgn_idx][end_idx] = 1
            adj_mat[end_idx][bgn_idx] = 1
        return adj_mat

    @property
    def id_adj_mat(self) -> np.ndarray:
        """Identity + Adjacency Matrix"""
        return self.adj_mat + np.eye(self.num_atoms)

    @property
    def norm_id_adj_mat(self) -> np.ndarray:
        """Spectral Normalized Identity Adjacency Matrix"""
        id_adj_mat = self.id_adj_mat
        norm_id_adj_mat = self._normalize_adj_mat(id_adj_mat)
        return norm_id_adj_mat

    @property
    def norm_laplacian_mat(self) -> np.ndarray:
        """Spectral Normalized Laplacian Matrix"""
        norm_adj_mat = self._normalize_adj_mat(self.adj_mat)
        norm_laplacian = np.eye(self.num_atoms) - norm_adj_mat
        return norm_laplacian

    @staticmethod
    def _normalize_adj_mat(adj_mat: Union[list, np.ndarray]) -> np.ndarray:
        """Helper function for normalizing adjacency matrices"""
        if isinstance(adj_mat, list):
            adj_mat = np.array(adj_mat)

        num_nodes = len(adj_mat)
        # Get vector of node degree
        node_deg = np.sum(adj_mat, axis=0)
        # Rescale node degree
        node_deg = 1 / np.sqrt(node_deg)
        # Convert to diag matrix of node degree
        node_deg = np.eye(num_nodes) * node_deg

        return np.matmul(np.matmul(node_deg, adj_mat), node_deg)


class Nodes:
    """Nodes class for MolGraph"""

    def __init__(self, mol: oechem.OEMol):
        """
        mol (oechem.OEMol): OEMol object
        
        """
        self.node_feat = self._get_node_features(mol)

    def _get_node_features(self, mol: oechem.OEMol) -> np.ndarray:
        node_feat = [self._featurize_atom(a) for a in mol.GetAtoms()]
        node_feat = np.array(node_feat)
        return node_feat

    @staticmethod
    def _featurize_atom(atom: oechem.OEAtomBase) -> np.ndarray:
        """Get atom feature vector
        
        atom (oechem.OEAtomBase): OEAtomBase object
        
        """
        results = (
            Nodes.one_of_k_encoding(
                oechem.OEGetAtomicSymbol(atom.GetAtomicNum()),
                [
                    "C",
                    "N",
                    "O",
                    "S",
                    "F",
                    "Si",
                    "P",
                    "Cl",
                    "Br",
                    "Mg",
                    "Na",
                    "Ca",
                    "Fe",
                    "As",
                    "Al",
                    "I",
                    "B",
                    "V",
                    "K",
                    "Tl",
                    "Yb",
                    "Sb",
                    "Sn",
                    "Ag",
                    "Pd",
                    "Co",
                    "Se",
                    "Ti",
                    "Zn",
                    "H",  # H?
                    "Li",
                    "Ge",
                    "Cu",
                    "Au",
                    "Ni",
                    "Cd",
                    "In",
                    "Mn",
                    "Zr",
                    "Cr",
                    "Pt",
                    "Hg",
                    "Pb",
                    "Unknown",
                ], unk=True
            )
            + Nodes.one_of_k_encoding(
                atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            )
            + Nodes.one_of_k_encoding(
                atom.GetImplicitHCount(), [0, 1, 2, 3, 4, 5, 6], unk=True
            )
            + [atom.GetFormalCharge()]
            + Nodes.one_of_k_encoding(
                atom.GetHyb(),
                [
                    oechem.OEHybridization_Unknown,
                    oechem.OEHybridization_sp,
                    oechem.OEHybridization_sp2,
                    oechem.OEHybridization_sp3,
                    oechem.OEHybridization_sp3d,
                    oechem.OEHybridization_sp3d2,
                ],
            )
            + [atom.IsAromatic()]
            + Nodes.one_of_k_encoding(
                atom.GetExplicitHCount(), [0, 1, 2, 3, 4], unk=True
            ),
        )
        return np.array(results)

    @staticmethod
    def one_of_k_encoding(x: Any, allowable_set: list, unk: bool=False) -> list:
        """Helper function for one hot encoding."""
        if x not in allowable_set:
            if unk:
                x = allowable_set[-1]
            else:
                raise Exception(
                    "input {0} not in allowable set{1}:".format(x, allowable_set)
                )
        return list(map(lambda s: x == s, allowable_set))


class MolGraph(Adjacency, Nodes):
    """ The graph corresponding to a molecule"""

    def __init__(self, smiles: str, explicit_H: Optional[str]=None):
        """ Create graph corresponding to a specified molecule.
        
        smiles (str): SMILES of molecule
        
        """
        self.mol = oechem.OEMol()
        oechem.OESmilesToMol(self.mol, smiles)
        self._process_mol(self.mol, explicit_H=explicit_H)

        Adjacency.__init__(self, self.mol)
        Nodes.__init__(self, self.mol)

    @staticmethod
    def _process_mol(mol: oechem.OEMol, explicit_H: Optional[str] = None):
        if explicit_H == 'all':
            oechem.OEAddExplicitHydrogens(mol)
        elif explicit_H == 'polar':
            oechem.OESuppressHydrogens(mol, explicit_H)
        elif explicit_H is None:
            oechem.OESuppressHydrogens(mol)
        else:
            raise ValueError
        oechem.OEAssignAromaticFlags(mol)
        oechem.OEAssignHybridization(mol)
        oechem.OEAssignFormalCharges(mol)
        mol.Sweep()

