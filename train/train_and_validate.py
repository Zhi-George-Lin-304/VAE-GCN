import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np

# Main Training and Validation Function
def train_and_validate(gcn_model, mlp_model, train_loader, val_loader, optimizer, epochs, low_energy_weight=10.0):
    train_mse_list = []
    val_mse_list = []

    for epoch in range(epochs):
        gcn_model.train()
        mlp_model.train()
        train_loss = 0.0

        for A, X, y, num_atoms in train_loader:
            A, X, y, num_atoms = A.to(device), X.to(device), y.to(device), num_atoms.to(device)
            optimizer.zero_grad()
            
            # GCN Embedding
            embeddings = gcn_model(A, X, num_atoms)
            
            # Prediction from MLP
            y_pred = mlp_model(embeddings)
            
            # Loss and Backpropagation
            loss = weighted_mse_loss(y_pred.squeeze(), y, low_energy_weight)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_mse_list.append(train_loss / len(train_loader))

        # Validation Phase
        gcn_model.eval()
        mlp_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for A, X, y, num_atoms in val_loader:
                A, X, y, num_atoms = A.to(device), X.to(device), y.to(device), num_atoms.to(device)
                embeddings = gcn_model(A, X, num_atoms)
                y_pred = mlp_model(embeddings)
                loss = weighted_mse_loss(y_pred.squeeze(), y, low_energy_weight)
                val_loss += loss.item()

        val_mse_list.append(val_loss / len(val_loader))
        scheduler.step(val_loss / len(val_loader))  # Update scheduler
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}")

    return train_mse_list, val_mse_list
