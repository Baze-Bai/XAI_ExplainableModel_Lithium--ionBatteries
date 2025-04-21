import  os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def Data_Preprocess(EIS_path = 'EIS_data/', Capacity_path = 'Capacity/'):
    """
    Preprocess EIS (Electrochemical Impedance Spectroscopy) and Capacity data for battery analysis.
    
    Args:
        EIS_path (str): Path to directory containing EIS data files
        Capacity_path (str): Path to directory containing Capacity data files
        
    Returns:
        tuple: Training, testing, and validation data and labels
            - Train_EIS: Training EIS data
            - Train_Capacity: Training capacity data (labels)
            - Test_EIS: Test EIS data
            - Test_Capacity: Test capacity data (labels)
            - Val_EIS: Validation EIS data
            - Val_Capacity: Validation capacity data (labels)
    """
    # Lists of file names for EIS and Capacity data across different temperatures
    EIS_files = ["EIS_state_I_25C01.txt","EIS_state_I_25C02.txt","EIS_state_I_25C03.txt","EIS_state_I_25C04.txt","EIS_state_I_25C05.txt","EIS_state_I_25C06.txt","EIS_state_I_25C07.txt","EIS_state_I_25C08.txt","EIS_state_I_35C01.txt","EIS_state_I_35C02.txt","EIS_state_I_45C01.txt","EIS_state_I_45C02.txt"]
    Capacity_files = ["Data_Capacity_25C01.txt","Data_Capacity_25C02.txt","Data_Capacity_25C03.txt","Data_Capacity_25C04.txt","Data_Capacity_25C05.txt","Data_Capacity_25C06.txt","Data_Capacity_25C07.txt","Data_Capacity_25C08.txt","Data_Capacity_35C01.txt","Data_Capacity_35C02.txt","Data_Capacity_45C01.txt","Data_Capacity_45C02.txt"]

    # Load EIS data files
    EIS = []
    for i in range(len(EIS_files)):
        EIS.append(pd.DataFrame(np.loadtxt(EIS_path + EIS_files[i],comments='t',delimiter='\t')))

    # Load Capacity data files
    Capacity = []
    for i in range(len(Capacity_files)):
        Capacity.append(pd.DataFrame(np.loadtxt(Capacity_path + Capacity_files[i],comments='t',delimiter='\t')))

    # Remove the first 3 columns from EIS data
    for i in range(len(EIS)):
        EIS[i] = EIS[i].iloc[:,3:]

    # Process EIS data by expanding every 60 rows into columns
    expanded_EIS = []
    for df in EIS:
        row_count = df.shape[0]
        col_count = df.shape[1]
        
        # Calculate how many groups can be formed (60 rows per group)
        group_count = row_count // 60
        
        # Create a new DataFrame to store the expanded data
        expanded_df = pd.DataFrame()
        
        for group in range(group_count):
            start_row = group * 60
            end_row = start_row + 60
            group_data = df.iloc[start_row:end_row, :]
            
            # Iterate through each column in the original DataFrame
            for col in range(col_count):
                # Get data from current column and expand into 60 columns
                col_data = group_data.iloc[:, col].values
                col_names = [f"{df.columns[col]}_{j+1}" for j in range(60)]
                
                # Add the expanded column data to the new DataFrame
                for j, name in enumerate(col_names):
                    expanded_df.loc[group, name] = col_data[j]
        
        expanded_EIS.append(expanded_df)
        
    # Normalize each capacity dataset
    normalized_Capacity = []
    for df in Capacity:
        # For single column data, normalize the entire dataframe
        data = df.values.flatten()  # Convert DataFrame to 1D array
        
        # Calculate min and max values
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Normalize to max value (not [0,1] range, just dividing by max)
        if max_val > min_val:  # Avoid division by zero
            normalized_data = data / max_val
        else:
            normalized_data = np.zeros_like(data)
        
        # Create new DataFrame with same index and column names
        normalized_df = pd.DataFrame(normalized_data, index=df.index, columns=df.columns)
        normalized_Capacity.append(normalized_df)
            
    # Select specific indices for the dataset
    l = [0,4,5,8,9,10,11]
    EIS_data = [expanded_EIS[i] for i in l]
    Capacity_data = [normalized_Capacity[i] for i in l]

    # Ensure EIS and Capacity data have matching lengths
    for i in range(len(EIS_data)):
        if len(EIS_data[i]) != len(Capacity_data[i]):
            EIS_data[i] = EIS_data[i].iloc[:len(Capacity_data[i])]
                
    # Reshape EIS data into (n_samples, 4, 60) format
    EIS_data_reshaped = []
    for i in range(len(EIS_data)):
        n_rows = EIS_data[i].shape[0]
        EIS_data_reshaped.append(EIS_data[i].values.reshape(n_rows, 4, 60))
        
    EIS_data = EIS_data_reshaped
    
    # Split data into training, test, and validation sets
    Test_EIS = EIS_data.pop(2)
    Test_Capacity = Capacity_data.pop(2)

    Val_EIS = EIS_data.pop(4)
    Val_Capacity = Capacity_data.pop(4)

    # Concatenate remaining data for training
    Train_EIS = np.concatenate(EIS_data, axis=0)
    Train_Capacity = np.concatenate(Capacity_data, axis=0)
    
    return Train_EIS, Train_Capacity, Test_EIS, Test_Capacity, Val_EIS, Val_Capacity

