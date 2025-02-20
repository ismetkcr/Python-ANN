# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:08:53 2025

@author: ismt
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import os
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


def read_excel_file_and_plot(file_path, show=True):
    """
    Reads an Excel file, extracts 'pH' and 'base_flow' columns, and plots them in subplots.

    Parameters:
        file_path (str): Path to the Excel file.
        show (bool): Whether to display the plot or not.

    Returns:
        pd.DataFrame: A DataFrame with integer index and 'pH', 'base_flow' columns (no 'time' column).
    """
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Ensure 'time', 'pH', and 'base_flow' columns are present
    required_columns = ['time', 'pH', 'base_flow']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The Excel file must contain the following columns: {required_columns}")
    
    # Extract 'pH' and 'base_flow' columns
    pH_values = df["pH"]
    base_flow_values = df["base_flow"]
    time_values = df["time"]
    
    # Create a new DataFrame with 'pH' and 'base_flow' columns
    result_df = pd.DataFrame({
        'pH': pH_values,
        'base_flow': base_flow_values
    })
    
    # Reset the index to integers (0, 1, 2, ...)
    result_df.reset_index(drop=True, inplace=True)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot pH values
    ax1.plot(time_values, result_df['pH'], label="Square Signal pH Values", color='blue')
    ax1.set_title("pH Values Over Time")
    ax1.set_ylabel("pH")
    ax1.legend()
    ax1.grid(True)
    
    # Plot base_flow values
    ax2.plot(time_values, result_df['base_flow'], label="Square Signal Flow Rate", color='green')
    ax2.set_title("Base Flow Rate Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Base Flow")
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Show or close the plot based on 'show' parameter
    if show:
        plt.show()
    else:
        plt.close(fig)  # Prevents displaying the plot when show=False
    
    # Return the new DataFrame (without the 'time' column)
    return result_df


def normalize_dataframe(df, show=True):
    """
    Normalize all columns of the given DataFrame to the specified min and max values and plot the normalized values.

    Parameters:
        df (pd.DataFrame): Input DataFrame with numerical columns.
        show (bool): Whether to display the plot or not.

    Returns:
        pd.DataFrame: Normalized DataFrame with the same column names.
    """
    # Define the feature ranges for normalization
    feature_ranges = {'pH': (2.9392, 10.2864), 'base_flow': (0, 250)}
    
    # Create a copy of the DataFrame to avoid modifying the original
    normalized_df = df.copy()
    
    # Normalize each column
    for col, (min_val, max_val) in feature_ranges.items():
        normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
    
    # Create subplots for normalized values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot normalized pH values
    ax1.plot(normalized_df.index, normalized_df['pH'], label="Normalized pH Values", color='blue')
    ax1.set_title("Normalized pH Values Over Index")
    ax1.set_ylabel("Normalized pH")
    ax1.legend()
    ax1.grid(True)
    
    # Plot normalized base_flow values
    ax2.plot(normalized_df.index, normalized_df['base_flow'], label="Normalized Base Flow Rate", color='green')
    ax2.set_title("Normalized Base Flow Rate Over Index")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Normalized Base Flow")
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Show or close the plot based on 'show' parameter
    if show:
        plt.show()
    else:
        plt.close(fig)  # Prevents displaying the plot when show=False
    
    # Return the normalized DataFrame
    return normalized_df


def make_multivariate_windows(data, window_size=10, horizon=1):
    windows = []
    labels = []
    for i in range(window_size, len(data) - horizon + 1):
        # Get window of y values: y(i-10) to y(i-1) (y[0] dan y[9]) dahil y[0:10], y[1, 11], y[2,12]...
        y_window = data.iloc[i-window_size:i]["pH"].values  # Shape: (window_size,)
        
        # Get window of u values: u(i-10+1) to u(i+1)e kadar  u[1] den u[11]e kadar
        u_window = data.iloc[i-window_size+1:i+1]["base_flow"].values  # Shape: (window_size,)
        
        # Combine y and u into a single window
        window = np.column_stack((y_window, u_window))  # Shape: (window_size, 2)
        
        # Get label (y(t))
        label = data.iloc[i:i+horizon]["pH"].values
        windows.append(window)
        labels.append(label)
        
    return np.array(windows), np.array(labels)

#make train test splits
def make_train_test_splits(windows, labels, test_split=0.1):
    split_size=int(len(windows)*(1-test_split))
    train_windows=windows[:split_size]
    train_labels=labels[:split_size]
    test_windows=windows[split_size:]
    test_labels=labels[split_size:]
    
    return train_windows, test_windows, train_labels, test_labels

def mean_absolute_scaled_error(y_true, y_pred):
    """
    Implement MASE (Mean Absolute Scaled Error)
    """
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Calculate MAE of naive forecast (using previous value)
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))
    
    # Avoid division by zero
    if mae_naive_no_season == 0:
        return tf.constant(0.0, dtype=tf.float32)
    
    return mae / mae_naive_no_season

def evaluate_preds(y_true, y_pred):
    """
    Evaluate predictions using various metrics.
    """
    y_true = tf.squeeze(tf.cast(y_true, dtype=tf.float32))
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    print(f"y_true shape is {y_true.shape}\n y_pred shape is {y_pred.shape}")
    
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(tf.reduce_mean(mse))
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)
    
    return {
        "mae": float(tf.reduce_mean(mae).numpy()),
        "mse": float(tf.reduce_mean(mse).numpy()),
        "rmse": float(rmse.numpy()),
        "mape": float(tf.reduce_mean(mape).numpy()),
        "mase": float(mase.numpy())
    }


def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val




def predict_1d(model, test_windows_1d, test_labels, test_u, window_size=10, 
                prediction_horizon=50, start_index=0, info=True):
    last_window_flat = test_windows_1d[start_index]
    last_window = last_window_flat.reshape(2, window_size).T
    
    test_labels_horizon = test_labels[start_index:start_index + prediction_horizon]
    test_u_horizon = test_u[start_index:start_index + prediction_horizon]
    
    predictions = []
    current_window = last_window.copy()
    
    for u in test_u_horizon:
        input_data = current_window.T.reshape(1, -1)
        y_pred = model.predict(input_data, verbose=0)[0, 0]
        predictions.append(y_pred)
        
        new_y = np.concatenate([current_window[1:, 0], [y_pred]])
        new_u = np.concatenate([current_window[1:, 1], [u]])
        current_window = np.column_stack((new_y, new_u))
    
    predictions = np.array(predictions)
    
    # Denormalize values
    predictions_denorm = denormalize(predictions, 2.9392, 10.2864)
    #predictions_denorm = denormalize(predictions, 0, 14)
    test_labels_horizon_denorm = denormalize(test_labels_horizon, 2.9392, 10.2864)
    #test_labels_horizon_denorm = denormalize(test_labels_horizon, 0, 14)
    test_u_horizon_denorm = denormalize(test_u_horizon, 0, 250)
    
    results = evaluate_preds(test_labels_horizon_denorm, predictions_denorm)
    
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(test_labels_horizon_denorm, label='True y (Denorm)')
    axs[0].plot(predictions_denorm, label='Predicted y (Denorm)', linestyle='dashed')
    axs[0].legend()
    axs[0].set_title(f"1D Model Predictions (Denormalized, Horizon = {prediction_horizon})")
    
    axs[1].plot(test_u_horizon_denorm, label='Test u (Denorm)', color='orange')
    axs[1].legend()
    axs[1].set_title("Input u over Prediction Horizon (Denormalized)")
    
    plt.tight_layout()
    plt.show()
    
    return predictions_denorm, results


def predict_2d(model, test_windows_2d, test_labels, test_u, window_size=10, 
                prediction_horizon=50, start_index=0, info=True):
    last_window = test_windows_2d[start_index]
    
    test_labels_horizon = test_labels[start_index:start_index + prediction_horizon]
    test_u_horizon = test_u[start_index:start_index + prediction_horizon]
    
    predictions = []
    current_window = last_window.copy()
    
    for u in test_u_horizon:
        input_data = current_window[np.newaxis, ...]
        y_pred = model.predict(input_data, verbose=0)[0, 0]
        predictions.append(y_pred)
        
        new_y = np.concatenate([current_window[1:, 0], [y_pred]])
        new_u = np.concatenate([current_window[1:, 1], [u]])
        current_window = np.column_stack((new_y, new_u))
    
    predictions = np.array(predictions)
    
    # Denormalize values
    predictions_denorm = denormalize(predictions, 2.9392, 10.2864)
    #predictions_denorm = denormalize(predictions, 0, 14)
    test_labels_horizon_denorm = denormalize(test_labels_horizon, 2.9392, 10.2864)
    #test_labels_horizon_denorm = denormalize(test_labels_horizon, 0, 14)
    test_u_horizon_denorm = denormalize(test_u_horizon, 0, 250)
    
    results = evaluate_preds(test_labels_horizon_denorm, predictions_denorm)
    
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(test_labels_horizon_denorm, label='True y (Denorm)')
    axs[0].plot(predictions_denorm, label='Predicted y (Denorm)', linestyle='dashed')
    axs[0].legend()
    axs[0].set_title(f"2D Model Predictions (Denormalized, Horizon = {prediction_horizon})")
    
    axs[1].plot(test_u_horizon_denorm, label='Test u (Denorm)', color='orange')
    axs[1].legend()
    axs[1].set_title("Input u over Prediction Horizon (Denormalized)")
    
    plt.tight_layout()
    plt.show()
    
    return predictions_denorm, results


# NBeatsBlock custom layer with serialization support
class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self, input_size: int, theta_size: int,
                 horizon: int, n_neurons: int, n_layers: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        
        # Layer definitions
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") 
                      for _ in range(n_layers)]
        self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)
        return theta[:, :self.input_size], theta[:, -self.horizon:]

    def get_config(self):
        # Essential for model serialization/deserialization
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "theta_size": self.theta_size,
            "horizon": self.horizon,
            "n_neurons": self.n_neurons,
            "n_layers": self.n_layers
        })
        return config
    
from tensorflow.keras import layers
#combines different models to predict common goal

def predict_ensemble(ensemble_models, test_windows_1d, test_labels, test_u, window_size=10,
                     prediction_horizon=50, start_index=0, aggregation='mean', info=True):
    # Initialize starting window
    last_window_flat = test_windows_1d[start_index]
    last_window = last_window_flat.reshape(2, window_size).T  # Shape: (window_size, 2)
    
    # Extract ground truth and inputs for the horizon
    test_labels_horizon = test_labels[start_index:start_index + prediction_horizon]
    test_u_horizon = test_u[start_index:start_index + prediction_horizon]
    
    # Store predictions from each model
    all_predictions = []
    
    # Make predictions with each model in the ensemble
    for model in ensemble_models:
        current_window = last_window.copy()
        model_predictions = []
        
        for u in test_u_horizon:
            # Prepare input data
            input_data = current_window.T.reshape(1, -1)
            
            # Predict next step
            y_pred = model.predict(input_data, verbose=0)[0, 0]  # Assumes 1-step output
            model_predictions.append(y_pred)
            
            # Update window with prediction and new u
            new_y = np.concatenate([current_window[1:, 0], [y_pred]])
            new_u = np.concatenate([current_window[1:, 1], [u]])
            current_window = np.column_stack((new_y, new_u))
        
        all_predictions.append(model_predictions)
    
    # Aggregate predictions across models
    all_predictions = np.array(all_predictions)
    if aggregation == 'mean':
        ensemble_predictions = np.mean(all_predictions, axis=0)
    elif aggregation == 'median':
        ensemble_predictions = np.median(all_predictions, axis=0)
    else:
        raise ValueError("Use 'mean' or 'median' aggregation")
    
    # Denormalize predictions and labels
    predictions_denorm = denormalize(ensemble_predictions, 0, 14)
    test_labels_horizon_denorm = denormalize(test_labels_horizon, 0, 14)
    test_u_horizon_denorm = denormalize(test_u_horizon, 0, 250)
    
    # Evaluate performance
    results = evaluate_preds(test_labels_horizon_denorm, predictions_denorm)
    
    # Plot results
    
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(test_labels_horizon_denorm, label='True Values')
    axs[0].plot(predictions_denorm, label=f'Ensemble ({aggregation}) Predictions', ls='--')
    axs[0].legend()
    axs[0].set_title(f"Ensemble Predictions vs True Values (Horizon={prediction_horizon})")
    
    axs[1].plot(test_u_horizon_denorm, label='Input Sequence (u)', color='orange')
    axs[1].legend()
    axs[1].set_title("Inputs Over Prediction Horizon")
    
    plt.tight_layout()
    plt.show()
    
    return predictions_denorm, results
