import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np

# Test set evaluation
def evaluate_test_set(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for A, X, y, num_atoms in test_loader:
            A, X, y, num_atoms = A.to(device), X.to(device), y.to(device), num_atoms.to(device)
            predictions = model(A, X, num_atoms).squeeze()
            y_true.extend(y.view(-1).cpu().numpy())  # Move to CPU for numpy conversion
            y_pred.extend(predictions.view(-1).cpu().numpy())  # Move to CPU for numpy conversion

    return np.array(y_true), np.array(y_pred)
