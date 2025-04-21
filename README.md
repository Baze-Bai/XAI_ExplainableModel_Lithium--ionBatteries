# Battery Capacity Prediction with Multi-Source 1D CNN

This project uses deep learning techniques to predict battery capacity based on Electrochemical Impedance Spectroscopy (EIS) data. It implements a 1D Convolutional Neural Network (CNN) architecture that processes multiple input sources simultaneously.

## Project Overview

The project analyzes battery performance using EIS data collected at different temperatures (25°C, 35°C, and 45°C). The goal is to predict battery capacity, which serves as an indicator of battery health and remaining useful life.

## Features

- Multi-source 1D CNN architecture for processing EIS data
- Data preprocessing pipeline for EIS and capacity measurements
- Custom dataset classes for handling multi-source time series data
- Integrated Gradients implementation for model explainability
- Comprehensive evaluation metrics including MSE and R²

## Directory Structure

```
.
├── Scripts/
│   ├── CNN_Class.py       # CNN model architecture
│   ├── Dataset_Class.py   # Custom dataset classes
│   ├── Data_Preprocess.py # Data preprocessing functions
│   └── Multi_Func.py      # Training, evaluation, and utility functions
├── EIS_data/              # EIS measurement data files
├── Capacity/              # Battery capacity data files
└── README.md              # This file
```

## Model Architecture

The model uses a multi-source 1D CNN architecture:
- Processes 4 input channels (sources) simultaneously
- Each channel has an identical branch of convolutional layers
- Results from each branch are concatenated and fed to fully connected layers
- Output is a single value representing the predicted battery capacity

## Data Processing

The data processing pipeline:
1. Loads EIS data files and capacity measurements
2. Preprocesses EIS data by reshaping into a format suitable for CNN (samples, channels, sequence length)
3. Normalizes capacity data
4. Splits data into training, validation, and test sets

## How to Use

### Prerequisites

```
python 3.8+
torch
numpy
pandas
matplotlib
scikit-learn
captum
scipy
```

### Training a Model

```python
from Scripts.Data_Preprocess import Data_Preprocess
from Scripts.CNN_Class import MultiSource1DCNN
from Scripts.Dataset_Class import create_dataloaders_from_separate_sets
from Scripts.Multi_Func import train_model, evaluate_regression

# Load and preprocess data
train_data, train_labels, test_data, test_labels, val_data, val_labels = Data_Preprocess()

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders_from_separate_sets(
    train_data, train_labels, val_data, val_labels, test_data, test_labels,
    is_classification=False, batch_size=32
)

# Initialize model
model = MultiSource1DCNN(
    seq_len=60,
    num_sources=4,
    conv_channels=(16, 32, 64),
    kernel_sizes=(3, 3, 3),
    fc_hidden=128,
    num_classes=1  # Regression task
)

# Train model
train_losses, val_losses, model = train_model(
    model, train_loader, val_loader,
    criterion=None, optimizer=None,  # Will be defined inside function
    num_epochs=1000, device='cuda', learning_rate=0.0001
)

# Evaluate model
mse, r2 = evaluate_regression(model, test_loader, device='cuda')
```

### Model Explainability

The project includes Integrated Gradients implementation for explaining model predictions:

```python
from Scripts.Multi_Func import intgrads_map

# Calculate attributions
attributions, delta = intgrads_map(
    model=model,
    inputs=input_batch,
    steps=200
)
```

## Running the Project

Follow these steps to run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/battery-capacity-prediction.git
cd battery-capacity-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install dependencies manually:

```bash
pip install torch numpy pandas matplotlib scikit-learn captum scipy
```

### 3. Prepare Data

Ensure your data is organized in the following structure:
- EIS data files in `EIS_data/` directory
- Capacity data files in `Capacity/` directory

### 4. Run Training Script

```bash
python main.py
```

Alternatively, you can create your own script using the code examples provided in the "Training a Model" section.

### 5. Evaluate and Visualize Results

The training script automatically:
- Saves the best model to `best_model.pt`
- Generates loss curves in `loss_curve.png`
- Displays MSE and R² metrics for the test set

### 6. Using a Trained Model for Predictions

```python
import torch
from Scripts.CNN_Class import MultiSource1DCNN

# Load model architecture
model = MultiSource1DCNN(
    seq_len=60,
    num_sources=4,
    conv_channels=(16, 32, 64),
    kernel_sizes=(3, 3, 3),
    fc_hidden=128,
    num_classes=1
)

# Load trained weights
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(input_data)
```

## Results

The model performance is evaluated using:
- Mean Squared Error (MSE)
- R² score

The training process saves the best model based on validation loss, and generates a loss curve plot to visualize training progress.

## License

This project is open source and available under the [MIT License](LICENSE). 