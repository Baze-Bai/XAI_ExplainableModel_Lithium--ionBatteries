import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class CustomMultiSourceDataset(Dataset):
    """Dataset class for loading existing dataframe data with shape [n,4,60]"""
    
    def __init__(self, dataframe, labels, is_classification=True):
        """
        Initialize the dataset
        
        Parameters:
        - dataframe: Data with shape (n, 4, 60), can be numpy array or pandas DataFrame
        - labels: Required, shape should be (n,) or (n, 1)
        - is_classification: Boolean, indicates whether this is a classification or regression task
        """
        # Ensure data is a numpy array
        if isinstance(dataframe, pd.DataFrame):
            self.data = dataframe.values.astype(np.float32)
        else:
            self.data = np.array(dataframe, dtype=np.float32)
        
        # Check data shape
        assert len(self.data.shape) == 3, "Data should be a 3D array with shape (n, 4, 60)"
        assert self.data.shape[1] == 4, "The second dimension should have 4 source signals"
        assert self.data.shape[2] == 60, "The third dimension should have 60 time points"
        
        # Convert data to torch tensor
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
        # Process labels
        # Ensure labels have the correct shape
        if isinstance(labels, pd.Series) or isinstance(labels, pd.DataFrame):
            labels = labels.values
        
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        
        # Convert labels based on task type
        if is_classification:
            # Convert classification labels to integer type
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            # Keep regression labels as float
            self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset by index"""
        return self.data[idx], self.labels[idx]

# Example usage
def create_dataloader(dataframe, labels, is_classification=True, 
                     batch_size=32, shuffle=True, random_state=42):
    """
    Create a single dataloader from data
    
    Parameters:
    - dataframe: Data with shape (n, 4, 60)
    - labels: Required, shape should be (n,) or (n, 1)
    - is_classification: Boolean, indicates whether this is a classification or regression task
    - batch_size: Batch size
    - shuffle: Whether to shuffle the data
    - random_state: Random seed
    
    Returns:
    - dataloader: A dataloader containing all data
    """
    # Create dataset
    dataset = CustomMultiSourceDataset(dataframe, labels, is_classification)
    
    # Set random seed
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def create_dataloaders_from_separate_sets(train_data, train_labels, 
                                         val_data, val_labels, 
                                         test_data, test_labels,
                                         is_classification=True,
                                         batch_size=32, 
                                         shuffle_train=True,
                                         shuffle_val=False,
                                         shuffle_test=False, 
                                         random_state=42):
    """
    Create train, validation, and test dataloaders from three separate datasets
    
    Parameters:
    - train_data: Training data, shape (n_train, 4, 60)
    - train_labels: Training labels, shape (n_train,) or (n_train, 1)
    - val_data: Validation data, shape (n_val, 4, 60)
    - val_labels: Validation labels, shape (n_val,) or (n_val, 1)
    - test_data: Test data, shape (n_test, 4, 60)
    - test_labels: Test labels, shape (n_test,) or (n_test, 1)
    - is_classification: Boolean, indicates whether this is a classification or regression task
    - batch_size: Batch size
    - shuffle_train: Whether to shuffle training data
    - shuffle_val: Whether to shuffle validation data
    - shuffle_test: Whether to shuffle test data
    - random_state: Random seed
    
    Returns:
    - train_loader: Training dataloader
    - val_loader: Validation dataloader
    - test_loader: Test dataloader
    """
    # Set random seed
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Create training dataloader
    train_dataset = CustomMultiSourceDataset(train_data, train_labels, is_classification)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    
    # Create validation dataloader
    val_dataset = CustomMultiSourceDataset(val_data, val_labels, is_classification)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_val)
    
    # Create test dataloader
    test_dataset = CustomMultiSourceDataset(test_data, test_labels, is_classification)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)
    
    return train_loader, val_loader, test_loader 