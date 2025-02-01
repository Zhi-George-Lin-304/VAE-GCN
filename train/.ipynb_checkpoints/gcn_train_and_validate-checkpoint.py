import torch
import torch.nn as nn

# Main Training and Validation Function for GCN with 5-Fold Cross-Validation
def train_and_validate(gcn_model, train_loader, val_loader, optimizer, epochs):
    train_mse_list = []
    val_mse_list = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for epoch in range(epochs):
        gcn_model.train()
        train_loss = 0.0

        for A, X, y, num_atoms in train_loader:
            A, X, y, num_atoms = A.to(device), X.to(device), y.to(device), num_atoms.to(device)
            optimizer.zero_grad()

            # GCN Embedding
            embeddings = gcn_model(A, X, num_atoms)

            # Loss and Backpropagation
            loss = nn.MSELoss()(embeddings.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_mse_list.append(train_loss / len(train_loader))

        # Validation Phase
        gcn_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for A, X, y, num_atoms in val_loader:
                A, X, y, num_atoms = A.to(device), X.to(device), y.to(device), num_atoms.to(device)
                embeddings = gcn_model(A, X, num_atoms)
                loss = nn.MSELoss()(embeddings.squeeze(), y)
                val_loss += loss.item()

        val_mse_list.append(val_loss / len(val_loader))

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}")

    return train_mse_list, val_mse_list
