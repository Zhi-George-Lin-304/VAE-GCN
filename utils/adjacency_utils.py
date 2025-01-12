from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np

def generate_adjacency_matrix(mol):
    num_atoms = mol.GetNumAtoms()
    adj_matrix = np.zeros((num_atoms, num_atoms), dtype=np.float32)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()

        # Map bond types to numeric values
        if bond_type == Chem.rdchem.BondType.SINGLE:
            adj_matrix[i, j] = 1.0
            adj_matrix[j, i] = 1.0
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            adj_matrix[i, j] = 2.0
            adj_matrix[j, i] = 2.0
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            adj_matrix[i, j] = 1.5
            adj_matrix[j, i] = 1.5
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            adj_matrix[i, j] = 3.0
            adj_matrix[j, i] = 3.0

    np.fill_diagonal(adj_matrix, 1.0)  # Add self-loops
    return adj_matrix
