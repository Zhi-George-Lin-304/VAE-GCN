import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
import numpy as np

def evaluate_test_set(gcn_model, mlp_model, test_loader):
    gcn_model.eval()
    mlp_model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for A, X, y, num_atoms in test_loader:
            # Move data to the appropriate device
            A, X, y, num_atoms = A.to(device), X.to(device), y.to(device), num_atoms.to(device)
            
            # Forward pass through GCN model
            gcn_output = gcn_model(A, X, num_atoms)
            
            # Forward pass through MLP model
            predictions = mlp_model(gcn_output).squeeze()
            
            # Collect true and predicted values
            y_true.extend(y.view(-1).cpu().numpy())  # Move to CPU for numpy conversion
            y_pred.extend(predictions.view(-1).cpu().numpy())  # Move to CPU for numpy conversion

    return np.array(y_true), np.array(y_pred)

