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

# Main
if __name__ == "__main__":
    # Hyperparameters
    input_dim = 61  # Feature size
    hidden_dim_gcn = 100
    hidden_dim_mlp = 300
    output_dim = 1
    batch_size = 1
    epochs = 100
    learning_rate = 0.0001
    low_energy_weight =1.0

    # Dataset
    train_dataset = MolecularDataset("/home/george/TADF/gcn/training_set.csv", target_col="ST_split")
    val_dataset = MolecularDataset("/home/george/TADF/gcn/validation_set.csv", target_col="ST_split")
    test_dataset = MolecularDataset("/home/george/TADF/gcn/testing_set.csv", target_col="ST_split")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer, and Scheduler
    gcn_model = GCN(input_dim, hidden_dim_gcn).to(device)
    mlp_model = MLP(hidden_dim_gcn, hidden_dim_mlp, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(gcn_model.parameters()) + list(mlp_model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


    # Train and validate
    train_mse, val_mse = train_and_validate(gcn_model, mlp_model, train_loader, val_loader, optimizer, epochs, low_energy_weight)


    # Save MSE values to CSV
    mse_df = pd.DataFrame({
        "Epoch": list(range(1, epochs + 1)),
        "Train_MSE": train_mse,
        "Validation_MSE": val_mse
    })
    mse_csv_path = "mse_train_validation_gcn_nn_leaky.csv"
    mse_df.to_csv(mse_csv_path, index=False)
    print(f"MSE values saved to '{mse_csv_path}'")

    # Test set evaluation and R-squared
    y_true, y_pred = evaluate_test_set(model, test_loader)
    r2 = r2_score(y_true, y_pred)

    # Save the trained model
    torch.save(model.state_dict(), "gcn_nn_leaky.pth")
    print("Model saved to 'gcn_model_leaky_gpu.pth'")

    # Save MSE values and plots
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_mse, label="Train MSE")
    plt.plot(range(1, epochs + 1), val_mse, label="Validation MSE")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (eV$^{2}$)")
    plt.title("Train and Validation MSE Over Epochs")
    plt.legend()
    plt.savefig("train_val_mse_gcn_nn_leaky.png")
    print("Train and Validation MSE plot saved as 'train_val_mse_gcn_nn_leaky.png'")

    # Scatter plot for test set predictions
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred, alpha=0.7, label="Predicted vs Actual")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", linestyle="--", label="Diagonal")
    plt.xlabel("Actual Values (eV)")
    plt.ylabel("Predicted Values (eV)")
    plt.title("Testing Set Predictions")
    #plt.legend()
    plt.text(0.05, 0.95, f"$R^2$: {r2:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.savefig("test_predictions_gcn_nn_leaky.png")
    print("Test set predictions plot saved as 'test_predictions_gcn_nn_leaky.png'")


    # Save MSE values to CSV
    mse_df = pd.DataFrame({"Epoch": list(range(1, epochs + 1)), "MSE": mse_list})
    mse_csv_path = "mse_over_epochs_gcn_nn_leaky.csv"
    mse_df.to_csv(mse_csv_path, index=False)
    print(f"MSE values saved to '{mse_csv_path}'")

    # Save GCN embeddings
    # add_embeddings_to_dataset(gcn_model, train_loader, train_dataset, "train_set_with_embeddings.csv")
    # add_embeddings_to_dataset(gcn_model, val_loader, val_dataset, "val_set_with_embeddings.csv")
    # add_embeddings_to_dataset(gcn_model, test_loader, test_dataset, "test_set_with_embeddings.csv")

