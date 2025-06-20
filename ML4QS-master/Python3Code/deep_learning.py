import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import xgboost as xgb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("all_features.csv", engine='python')
# df = pd.read_csv("all_exercises_timeseries.csv", engine='python')
print(f"dataset used: all_features.csv")


# convert 'exercise' to categorical
df['exercise'] = df['exercise'].astype('category')
# convert 'exercise' to numerical codes
df['exercise_code'] = df['exercise'].cat.codes
# check the unique codes
# print("Unique exercise codes:", df['exercise_code'].unique())
# Unique exercises: ['burpees' 'crunches' 'jumping_jacks' 'plank' 'squats']



# Define features and labels
columns_to_drop=['exercise', 'participant' , 'dataset' , 'exercise_participant', 'gyro_x_mean', 'gyro_x_std', 'gyro_x_min', 'gyro_x_max', 'gyro_x_range',
       'gyro_x_median', 'gyro_x_q25', 'gyro_x_q75', 'gyro_x_iqr',
       'gyro_x_energy', 'gyro_x_rms', 'gyro_magnitude_mean',
       'gyro_magnitude_std', 'gyro_magnitude_max', 'exercise_code']
existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
X = df.drop(columns=existing_cols_to_drop, errors='ignore')

y = df['exercise_code']



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23, stratify=y)



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation, padding=(kernel_size - 1) * dilation,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: (batch, features, seq_len)
        y = self.network(x)
        y = y[:, :, -1]  # take last time step
        return self.linear(y)


def train_tcn_model(X_train, y_train, X_test, y_test, input_size, output_size, num_channels):
    """
    Train a Temporal Convolutional Network (TCN) model and evaluate its performance.
    
    Parameters:
    - X_train: Training feature set
    - y_train: Training labels
    - X_test: Test feature set
    - y_test: Test labels
    - input_size: Number of input features
    - output_size: Number of output classes
    - num_channels: List of channels for each TCN layer
    
    Returns:
    - accuracy: Accuracy of the model on the test set
    """
    
    # # Convert data to PyTorch tensors if available
    # if device.type == 'cuda':
    #     X_train = X_train.to(device)
    #     y_train = y_train.to(device)
    #     X_test = X_test.to(device)
    #     y_test = y_test.to(device)
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(2) 
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(2) 
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create the model
    model = TCN(input_size=input_size, output_size=output_size, num_channels=num_channels)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(10):  # Number of epochs can be adjusted
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test_tensor).float().mean().item()

    return accuracy


print("Training TCN model...")
print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
print(f"Test shape: {X_test.shape}, Test labels shape: {y_test.shape}")
print(type(X_train), type(y_train), type(X_test), type(y_test))

input_size = X_train.shape[1]
output_size = len(y.unique())
num_channels = [64, 64, 64]  # Example channel sizes for each TCN layer
accuracy = train_tcn_model(
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    input_size, 
    output_size, 
    num_channels
)
print(f"TCN Model Accuracy: {accuracy:.4f}")



# lightgbm model

def train_lightgbm_model(X_train, y_train, X_test, y_test, max_depth=10, eta=0.1, gamma=0.1, subsample=0.8, colsample_bytree=0.8):
    """
    Train a LightGBM model and evaluate its performance.
    
    Parameters:
    - X_train: Training feature set
    - y_train: Training labels
    - X_test: Test feature set
    - y_test: Test labels
    
    Returns:
    - accuracy: Accuracy of the model on the test set
    """
    
    # Convert data to LightGBM Dataset format
    train_data = xgb.DMatrix(X_train, label=y_train)
    test_data = xgb.DMatrix(X_test, label=y_test)

    # Set parameters for LightGBM
    params = {
        'objective': 'multi:softmax',
        'num_class': len(y.unique()),
        'eval_metric': 'mlogloss',
        'max_depth': max_depth,
        'eta': eta,
        'gamma': gamma,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'seed': 23
    }

    # Train the model
    model = xgb.train(params, train_data, num_boost_round=100)

    # Make predictions
    predictions = model.predict(test_data)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test.values)

    return accuracy

print("Training LightGBM model...")
accuracy_lightgbm = train_lightgbm_model(X_train, y_train, X_test, y_test)  
print(f"LightGBM Model Accuracy: {accuracy_lightgbm:.4f}")