import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def weighted_mse_loss(y_pred, y_true, low_energy_weight=10.0):
    weights = torch.where(y_true < 0.5, low_energy_weight, 1.0)  # Weight low-energy (<0.5) more heavily
    loss = weights * (y_pred - y_true) ** 2
    return loss.mean()

def train_one_epoch(gcn_model, mlp_model, train_loader, optimizer, low_energy_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    gcn_model.train()
    mlp_model.train()
    total_loss = 0.0

    for A, X, y, num_atoms in train_loader:
        A, X, y, num_atoms = A.to(device), X.to(device), y.to(device), num_atoms.to(device)
        optimizer.zero_grad()

        # GCN Embedding
        embeddings = gcn_model(A, X, num_atoms)

        # Prediction from MLP
        y_pred = mlp_model(embeddings)

        # Compute loss and backpropagation
        loss = weighted_mse_loss(y_pred.squeeze(), y, low_energy_weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate_one_epoch(gcn_model, mlp_model, val_loader, low_energy_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gcn_model.eval()
    mlp_model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for A, X, y, num_atoms in val_loader:
            A, X, y, num_atoms = A.to(device), X.to(device), y.to(device), num_atoms.to(device)

            # GCN Embedding
            embeddings = gcn_model(A, X, num_atoms)

            # Prediction from MLP
            y_pred = mlp_model(embeddings)

            # Compute loss
            loss = weighted_mse_loss(y_pred.squeeze(), y, low_energy_weight)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def save_embeddings_last_epoch(gcn_model, train_loader, dataset, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings_list = []
    smiles_list = []
    targets_list = []

    gcn_model.eval()
    with torch.no_grad():
        for batch in train_loader:
            A, X, num_atoms, targets, idx = batch  # idx contains dataset indices
            A, X, num_atoms, targets = A.to(device), X.to(device), num_atoms.to(device), targets.to(device)

            # Retrieve SMILES using dataset mapping
            smiles_batch = [dataset.smiles_dict[i.item()] for i in idx]
            smiles_list.extend(smiles_batch)

            # Generate embeddings
            embeddings = gcn_model(A, X, num_atoms)  # Use A (not A_normalized)
            embeddings_list.extend(embeddings.cpu().numpy())

            # Convert targets to NumPy and store them
            targets_list.extend(targets.cpu().numpy())

    # Convert to DataFrame
    embeddings_df = pd.DataFrame({
        "smiles": smiles_list,
        "E(S1)": [target[0] for target in targets_list],
        "E(T1)": [target[1] for target in targets_list],
        "ST_split": [target[2] for target in targets_list],
        "embeddings": embeddings_list
    })

    # Save the embeddings to CSV
    output_path = f"C:/Users/user/Downloads/VAE_GCN/VAE_GCN_results/embeddings/embeddings_fold_{fold}_final.csv"
    embeddings_df.to_csv(output_path, index=False)
    print(f"Embeddings for Fold {fold} saved to '{output_path}'")


def train_and_validate(gcn_model, mlp_model, train_loader, val_loader, optimizer, epochs, fold, low_energy_weight=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_mse_list = []
    val_mse_list = []

    for epoch in range(epochs):
        # Training Phase
        train_loss = train_one_epoch(gcn_model, mlp_model, train_loader, optimizer, low_energy_weight)
        train_mse_list.append(train_loss)

        # Validation Phase
        val_loss = validate_one_epoch(gcn_model, mlp_model, val_loader, low_energy_weight)
        val_mse_list.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save embeddings after the last epoch
    save_embeddings_last_epoch(gcn_model, train_loader, fold)

    return train_mse_list, val_mse_list

