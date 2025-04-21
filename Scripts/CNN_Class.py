import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSource1DCNN(nn.Module):
    """
    Multi-Source 1D CNN Architecture:
    Four input channels → Three convolutional layers (independent but shared structure for each channel) → Concatenation → Fully connected layers
    
    Input shape: (B, 4, 60) where B is batch size
    Output shape: (B, num_classes) for classification or (B, 1) for regression
    """
    def __init__(
        self,
        seq_len: int = 60,
        num_sources: int = 4,
        conv_channels=(16, 32, 64),      # Output channels for the three conv layers
        kernel_sizes=(3, 3, 3),          # Kernel sizes for the three conv layers
        fc_hidden: int = 128,
        num_classes: int = 1             # 1 for binary classification/regression; >1 for multi-class with softmax
    ):
        super().__init__()
        self.num_sources = num_sources
        self.seq_len = seq_len
        c1, c2, c3 = conv_channels
        k1, k2, k3 = kernel_sizes

        # ---- Convolutional stack for each input channel (shared structure) ----
        self.branch = nn.Sequential(
            nn.Conv1d(1,  c1, k1, padding=k1 // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, k2, padding=k2 // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(c2, c3, k3, padding=k3 // 2),
            nn.ReLU(inplace=True),
        )
        # Create num_sources copies of the branch using ModuleList
        self.branches = nn.ModuleList([self.branch for _ in range(num_sources)])

        # ---- Fully connected layers ----
        flat_dim = num_sources * c3 * seq_len   # If pooling is added, this dimension would need recalculation
        self.fc1 = nn.Linear(flat_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (B, 4, 60) where 4 channels are along dimension 1
            
        Returns:
            Output tensor of shape (B, num_classes)
        """
        assert x.size(1) == self.num_sources and x.size(2) == self.seq_len

        # Process each input channel through its own convolutional branch
        outs = []
        for i, branch in enumerate(self.branches):
            xi = x[:, i:i+1, :]          # (B, 1, 60)
            yi = branch(xi)              # (B, C3, 60)
            outs.append(yi)

        # Concatenate along channel dimension → (B, 4*C3, 60)
        y_cat = torch.cat(outs, dim=1)

        # Flatten and pass through fully connected layers
        y_flat = y_cat.view(x.size(0), -1)
        y = F.relu(self.fc1(y_flat))
        y = self.fc2(y)                 # For regression: direct output; For classification: add Sigmoid/Softmax
        return y
