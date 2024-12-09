import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import torch
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# Import necessary modules for model and time feature processing
from models import iTransformer
from utils.timefeatures import time_features

# Solve the problem of Chinese display in matplotlib plots
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def tslib_data_loader(window, length_size, batch_size, data, data_mark):
    """
    Data Loader Function for Time Series Data
    This function is designed to load and preprocess time series data for training models.
    It supports both multivariate-to-multivariate and univariate-to-univariate predictions.

    Parameters:
        window (int): The window size, determining the length of the input sequence.
        length_size (int): The length of the target sequence.
        batch_size (int): The batch size for training.
        data (ndarray): Input time series data.
        data_mark (ndarray): Markers for the input data to assist in training.

    Returns:
        dataloader (DataLoader): DataLoader object for batch processing.
        x_temp (Tensor): Processed input data.
        y_temp (Tensor): Processed target data.
        x_temp_mark (Tensor): Markers for input data.
        y_temp_mark (Tensor): Markers for target data.
    """
    # Define sequence length
    seq_len = window
    sequence_length = seq_len + length_size

    # Create sequences for data and markers
    result = np.array([data[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])
    result_mark = np.array([data_mark[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])

    # Split sequences into input (x) and target (y)
    x_temp = result[:, :-length_size]
    y_temp = result[:, -(length_size + int(window / 2)):]

    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)):]

    # Convert data to PyTorch tensors
    x_temp = torch.tensor(x_temp).type(torch.float32)
    x_temp_mark = torch.tensor(x_temp_mark).type(torch.float32)
    y_temp = torch.tensor(y_temp).type(torch.float32)
    y_temp_mark = torch.tensor(y_temp_mark).type(torch.float32)

    # Create DataLoader for batch processing
    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark

def model_train(net, train_loader, length_size, optimizer, criterion, num_epochs, device, print_train=False):
    """
    Train the model.

    Parameters:
        net (torch.nn.Module): Model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        length_size (int): Length of the target sequence.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (torch.nn.Module): Loss function.
        num_epochs (int): Number of epochs for training.
        device (torch.device): Device for training (CPU/GPU).
        print_train (bool): If True, prints training progress.

    Returns:
        net (torch.nn.Module): Trained model.
        train_loss (list): List of average training losses.
        best_epoch (int): The epoch with the best loss.
    """
    train_loss = []  # Record average training loss
    print_frequency = num_epochs / 20  # Calculate print frequency

    for epoch in range(num_epochs):
        total_train_loss = 0  # Initialize epoch loss
        net.train()  # Set model to training mode

        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(train_loader):
            datapoints, labels, datapoints_mark, labels_mark = (
                datapoints.to(device), labels.to(device), datapoints_mark.to(device), labels_mark.to(device))
            optimizer.zero_grad()  # Clear gradients
            preds = net(datapoints, datapoints_mark, labels, labels_mark, None).squeeze()  # Forward pass
            labels = labels[:, -length_size:].squeeze()
            loss = criterion(preds, labels)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            total_train_loss += loss.item()  # Accumulate loss

        avg_train_loss = total_train_loss / len(train_loader)  # Compute average loss
        train_loss.append(avg_train_loss)

        # Print training progress
        if print_train and (epoch + 1) % print_frequency == 0:
            print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

    return net, train_loss, epoch + 1
def model_train_val(net, train_loader, val_loader, length_size, optimizer, criterion, scheduler, num_epochs, device, early_patience=0.15, print_train=False):
    """
    Train the model with validation and apply early stopping.

    Parameters:
    - net (torch.nn.Module): The model to train.
    - train_loader (torch.utils.data.DataLoader): Training data loader.
    - val_loader (torch.utils.data.DataLoader): Validation data loader.
    - length_size (int): Length of the output sequence.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - criterion (torch.nn.Module): Loss function.
    - scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    - num_epochs (int): Total number of epochs.
    - device (torch.device): Device (CPU or GPU).
    - early_patience (float, optional): Patience for early stopping, default is 0.15 * num_epochs.
    - print_train (bool, optional): Whether to print training information. Default is False.

    Returns:
    - net (torch.nn.Module): Trained model.
    - train_loss (list): List of average training losses for each epoch.
    - val_loss (list): List of average validation losses for each epoch.
    - epoch (int): Epoch at which early stopping is triggered.
    """
    train_loss = []  # Record average training losses
    val_loss = []  # Record validation losses for early stopping
    print_frequency = num_epochs / 20  # Calculate print frequency

    early_patience_epochs = int(early_patience * num_epochs)  # Convert early stopping patience to epochs
    best_val_loss = float('inf')  # Initialize the best validation loss
    early_stop_counter = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        total_train_loss = 0  # Initialize epoch loss
        net.train()  # Set model to training mode

        for datapoints, labels, datapoints_mark, labels_mark in train_loader:
            datapoints, labels, datapoints_mark, labels_mark = (
                datapoints.to(device), labels.to(device), datapoints_mark.to(device), labels_mark.to(device)
            )
            optimizer.zero_grad()  # Clear gradients
            preds = net(datapoints, datapoints_mark, labels, labels_mark, None).squeeze()  # Forward pass
            labels = labels[:, -length_size:].squeeze()  # Extract last part of the target
            loss = criterion(preds, labels)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            total_train_loss += loss.item()  # Accumulate loss

        avg_train_loss = total_train_loss / len(train_loader)  # Calculate average loss
        train_loss.append(avg_train_loss)  # Append average loss

        # Validation loop
        with torch.no_grad():
            total_val_loss = 0
            for val_x, val_y, val_x_mark, val_y_mark in val_loader:
                val_x, val_y, val_x_mark, val_y_mark = (
                    val_x.to(device), val_y.to(device), val_x_mark.to(device), val_y_mark.to(device)
                )
                pred_val_y = net(val_x, val_x_mark, val_y, val_y_mark, None).squeeze()  # Forward pass
                val_y = val_y[:, -length_size:].squeeze()  # Extract last part of the target
                val_loss_batch = criterion(pred_val_y, val_y)  # Calculate loss
                total_val_loss += val_loss_batch.item()

            avg_val_loss = total_val_loss / len(val_loader)  # Calculate average validation loss
            val_loss.append(avg_val_loss)  # Append validation loss
            scheduler.step(avg_val_loss)  # Update learning rate based on validation loss

        # Print training information
        if print_train and (epoch + 1) % print_frequency == 0:
            print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0  # Reset counter
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_patience_epochs:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break  # Stop training

    net.train()  # Set model back to training mode
    return net, train_loss, val_loss, epoch + 1


def cal_eval(y_real, y_pred):
    """
    Evaluate prediction performance.

    Parameters:
    - y_real (numpy array): Actual target values from the test set.
    - y_pred (numpy array): Predicted values.

    Returns:
    - df_eval (pandas DataFrame): DataFrame containing evaluation metrics.
    """
    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()

    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred, squared=True)
    rmse = mean_squared_error(y_real, y_pred, squared=False)  # Root Mean Squared Error
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100  # Convert to percentage

    df_eval = pd.DataFrame({
        'R2': r2,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }, index=['Eval'])

    return df_eval


# Load data
df = pd.read_csv('data\\etth2.csv')
# Ensure the target variable is the last column for multivariate cases
data_dim = df[df.columns.drop('date')].shape[1]  # Number of variables
data_target = df['Target']  # Target variable for prediction
data = df[df.columns.drop('date')]  # Select all relevant data

# Process timestamps
df_stamp = df[['date']]
df_stamp['date'] = pd.to_datetime(df_stamp.date)
data_stamp = time_features(df_stamp, timeenc=1, freq='H')  # Encode time features based on hourly frequency

# Data normalization
scaler = MinMaxScaler()
data_inverse = scaler.fit_transform(np.array(data))

data_length = len(data_inverse)
train_set = 0.8  # Training set proportion

data_train = data_inverse[:int(train_set * data_length), :]  # Training data
data_train_mark = data_stamp[:int(train_set * data_length), :]  # Training data markers
data_test = data_inverse[int(train_set * data_length):, :]  # Test data
data_test_mark = data_stamp[int(train_set * data_length):, :]  # Test data markers

window = 10  # Input sequence length
length_size = 1  # Output sequence length
batch_size = 32

train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(
    window, length_size, batch_size, data_train, data_train_mark
)
test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(
    window, length_size, batch_size, data_test, data_test_mark
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 20
learning_rate = 0.0001
scheduler_patience = int(0.25 * num_epochs)  # Convert to integer
early_patience = 0.2  # Proportion of training iterations for early stopping

# Model configuration class
class Config:
    def __init__(self):
        self.seq_len = window
        self.label_len = int(window / 2)
        self.pred_len = length_size
        self.freq = 'h'
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.stop_ratio = early_patience
        self.dec_in = data_dim
        self.enc_in = data_dim
        self.c_out = 1
        self.d_model = 64
        self.n_heads = 8
        self.dropout = 0.1
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 64
        self.factor = 5
        self.activation = 'gelu'
        self.channel_independence = 0
        self.embed = 'timeF'
        self.output_attention = 0
        self.task_name = 'short_term_forecast'
        self.moving_avg = window - 1


config = Config()
net = iTransformer.Model(config).to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience, verbose=True)

# Train the model
trained_model, train_loss, final_epoch = model_train(
    net, train_loader, length_size, optimizer, criterion, num_epochs, device, print_train=True
)
trained_model.eval()  # Set the model to evaluation mode

# Perform predictions and adjust dimensions
pred = trained_model(x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device))
true = y_test[:, -length_size:, -1:].detach().cpu()
pred = pred.detach().cpu()

# Check and adjust dimensions of predictions and true values
print("Shape of true before adjustment:", true.shape)
print("Shape of pred before adjustment:", pred.shape)

# Adjust predictions and true values to be 2D arrays
true = true[:, :, -1]
pred = pred[:, :, -1]
print("Shape of pred after adjustment:", pred.shape)
print("Shape of true after adjustment:", true.shape)

# Update scaler for inverse transformation (specific to the target data)
y_data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))
pred_uninverse = scaler.inverse_transform(pred[:, -1:])  # For multi-step prediction, use the last column
true_uninverse = scaler.inverse_transform(true[:, -1:])

# Final true and predicted values after inverse transformation
true, pred = true_uninverse, pred_uninverse

# Evaluate the predictions
df_eval = cal_eval(true, pred)
print(df_eval)

# Plot predictions vs. true values
df_pred_true = pd.DataFrame({'Predict': pred.flatten(), 'Real': true.flatten()})
df_pred_true.plot(figsize=(10, 8))
plt.title('WTC-iTransformer Result')
plt.savefig('WTC-iTransformer_Comparison.png', dpi=1200)
plt.show()

# Save true and predicted values to a CSV file
result_df = pd.DataFrame({'True': true.flatten(), 'Predict': pred.flatten()})
result_df.to_csv('WTC-iTransformer_True_vs_Predict.csv', index=False, encoding='utf-8')
print('True and predicted values saved to WTC-iTransformer_True_vs_Predict.csv.')

# Plot scatter plot of true vs. predicted values
fig3 = plt.figure(figsize=(10, 8))
plt.scatter(true, pred, color='#6CA6CD', edgecolors=(0, 0, 0), s=80)
plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r-', lw=4)
plt.xlabel('True', fontsize='medium')
plt.ylabel('Predicted', fontsize='medium')
plt.title('WTC-iTransformer Scatter Plot', fontsize='large')
plt.grid()
plt.savefig('WTC-iTransformer_Scatter.png', dpi=1200)
plt.show()

# Plot histogram of prediction errors
fig4 = plt.figure(figsize=(10, 8))
errors = true - pred
plt.hist(errors, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel('Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('WTC-iTransformer Error Distribution', fontsize=14)
mean_error = np.mean(errors)
plt.axvline(mean_error, color='r', linestyle='dashed', linewidth=2, label=f'Mean Error = {mean_error:.4f}')
plt.legend()
plt.savefig('WTC-iTransformer_Error_Histogram.png', dpi=1200)
plt.show()
