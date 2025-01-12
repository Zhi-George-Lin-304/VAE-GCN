import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from utils.feature_utils import generate_feature_matrix
from utils.adjacency_utils import generate_adjacency_matrix

class MolecularDataset(Dataset):
    def __init__(self, csv_file, target_col, max_atoms=460):
        data = pd.read_csv(csv_file).dropna(subset=['SMILES'])
        data = data[data['SMILES'].apply(lambda x: isinstance(x, str))]

        self.smiles = []
        self.targets = []
        self.max_atoms = max_atoms

        for i, row in data.iterrows():
            smiles = row['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                self.smiles.append(smiles)
                self.targets.append(row[target_col])

        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        mol = Chem.MolFromSmiles(smiles)

        adj = generate_adjacency_matrix(mol)
        features = generate_feature_matrix(mol)

        padded_adj = np.zeros((self.max_atoms, self.max_atoms), dtype=np.float32)
        num_atoms = adj.shape[0]
        padded_adj[:num_atoms, :num_atoms] = adj

        adj_hat = padded_adj + np.eye(self.max_atoms, dtype=np.float32)
        D_hat_inv_sqrt = np.diag(np.power(np.sum(adj_hat, axis=1), -0.5, where=np.sum(adj_hat, axis=1) != 0))
        adj_normalized = np.matmul(np.matmul(D_hat_inv_sqrt, adj_hat), D_hat_inv_sqrt)

        padded_features = np.zeros((self.max_atoms, 61), dtype=np.float32)
        padded_features[:num_atoms, :] = features

        target = self.targets[idx]
        return (
            torch.tensor(adj_normalized, dtype=torch.float32),
            torch.tensor(padded_features, dtype=torch.float32),
            target,
            torch.tensor(num_atoms, dtype=torch.int64),
        )
