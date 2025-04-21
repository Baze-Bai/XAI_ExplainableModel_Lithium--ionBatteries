import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from Scripts.CNN_Class import MultiSource1DCNN
from Scripts.Dataset_Class import create_dataloaders_from_separate_sets
from captum.attr import IntegratedGradients

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000, device='cuda', learning_rate=0.0001):
    """
    Train a PyTorch model with validation monitoring.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer to use
        num_epochs: Number of training epochs
        device: Device to use (cuda/cpu)
        learning_rate: Learning rate for the optimizer
        
    Returns:
        tuple: Training losses, validation losses, and the trained model
    """
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Using MSE loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')   # To track the lowest validation loss so far

    for epoch in range(num_epochs):
        # ---- Training phase ---- #
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # ---- Validation phase ---- #
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} — '
            f'Train Loss: {epoch_train_loss:.4f}, '
            f'Val Loss:   {epoch_val_loss:.4f}')
        
        # ---- Save best model ---- #
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'Validation loss improved. Saving model (epoch {epoch+1}).')

    
    return train_losses, val_losses, model

# Evaluate regression model
def evaluate_regression(model, test_loader, device='cuda'):
    """
    Evaluate a regression model using MSE and R² metrics.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        device: Device to use (cuda/cpu)
        
    Returns:
        tuple: Average MSE and R² score
    """
    model.eval()
    mse_loss = nn.MSELoss()
    all_targets = []
    all_predictions = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = mse_loss(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            # Collect predictions and actual values for R² calculation
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
    
    avg_mse = running_loss / len(test_loader.dataset)
    # Calculate R² score
    r2 = r2_score(np.array(all_targets), np.array(all_predictions))
    
    print(f'MSE on test set: {avg_mse:.4f}')
    print(f'R2 score on test set: {r2:.4f}')
    return avg_mse, r2


def main():
    """
    Main function to demonstrate the complete workflow:
    1. Set parameters
    2. Prepare data
    3. Build model
    4. Train model
    5. Evaluate model
    6. Save model
    """
    # Parameter settings
    seq_len = 60
    num_sources = 4
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    
    # Check for available CUDA device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    # Create three separate datasets - training, validation, and test sets
    # Replace these with actual data and labels in real applications
    # Example data - replace with actual data in real applications
    train_data = np.random.randn(700, 4, 60).astype(np.float32)  # Training data
    train_labels = np.random.randn(700, 1).astype(np.float32)    # Training labels
    
    val_data = np.random.randn(150, 4, 60).astype(np.float32)    # Validation data
    val_labels = np.random.randn(150, 1).astype(np.float32)      # Validation labels
    
    test_data = np.random.randn(150, 4, 60).astype(np.float32)   # Test data
    test_labels = np.random.randn(150, 1).astype(np.float32)     # Test labels
    
    # Create data loaders for each dataset
    train_loader, val_loader, test_loader = create_dataloaders_from_separate_sets(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        test_data=test_data,
        test_labels=test_labels,
        is_classification=False,  # Set as regression task
        batch_size=batch_size,
        shuffle_train=True,
        shuffle_val=False,
        shuffle_test=False,
        random_state=42
    )
    
    # Build model
    model = MultiSource1DCNN(
        seq_len=seq_len,
        num_sources=num_sources,
        conv_channels=(16, 32, 64),
        kernel_sizes=(3, 3, 3),
        fc_hidden=128,
        num_classes=1  # Output dimension is 1 for regression task
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Using MSE loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=num_epochs, device=device
    )
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()
    
    # Evaluate model
    mse, r2 = evaluate_regression(model, test_loader, device)
    
    # Save model
    torch.save(model.state_dict(), 'cnn_model.pth')
    print("Model saved to 'cnn_model.pth'")

def intgrads_map(
    model: torch.nn.Module,
    inputs: torch.Tensor,            # shape: (B, 3, 60)
    target: int | list[int] | None = None,
    baseline: torch.Tensor | None = None,
    steps: int = 200,
) -> torch.Tensor:
    """
    Batch Integrated Gradients calculation.
    
    Args:
        model: Neural network model
        inputs: Input tensor of shape (B, 3, 60)
        target: For regression/binary classification, can be None;
                For multi-class classification:
                - target=int: use the same class for all samples
                - target=list[int] of length B: specify class for each sample
        baseline: If None, a zero tensor with same shape as inputs is used
        steps: Number of steps for the integral approximation
        
    Returns:
        tuple: Attribution values and convergence delta
            - attrs: Tensor of shape (B, 3, 60) with attribution values
            - delta: Tensor of shape (B,) with convergence error metrics
    """
    model.eval()
    ig = IntegratedGradients(model)

    # If no baseline provided, use a zero tensor
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    # Target handling: if a single int is provided, it's automatically broadcast to all batch items
    # If a list[int] is provided, Captum can attribute each sample by its respective index
    attrs, delta = ig.attribute(
        inputs,
        baselines=baseline,
        target=target,
        n_steps=steps,
        return_convergence_delta=True
    )
    
    # attrs: (B, 3, 60) - attribution values
    # delta: (B,) - optional convergence error metric

    return attrs.detach(), delta.detach()