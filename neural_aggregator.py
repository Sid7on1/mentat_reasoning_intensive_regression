import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from typing import List, Tuple, Dict
import logging
from logging import Logger
import json
import os

# Set up logging
logger: Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler('neural_aggregator.log')
handler.setFormatter(formatter)
logger.addHandler(handler)

class MLPAggregator(nn.Module):
    """
    A PyTorch module for aggregating multiple LLM rollouts into final predictions.

    Attributes:
    input_dim (int): The number of input features.
    hidden_dim (int): The number of hidden units in the MLP.
    output_dim (int): The number of output features.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initializes the MLPAggregator.

        Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of hidden units in the MLP.
        output_dim (int): The number of output features.
        """
        super(MLPAggregator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the MLPAggregator.

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class NeuralAggregator:
    """
    A class for aggregating multiple LLM rollouts into final predictions.

    Attributes:
    input_dim (int): The number of input features.
    hidden_dim (int): The number of hidden units in the MLP.
    output_dim (int): The number of output features.
    device (str): The device to use for computations.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: str = 'cpu'):
        """
        Initializes the NeuralAggregator.

        Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of hidden units in the MLP.
        output_dim (int): The number of output features.
        device (str): The device to use for computations.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.model = MLPAggregator(input_dim, hidden_dim, output_dim)
        self.model.to(device)

    def train_aggregator(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001) -> None:
        """
        Trains the MLPAggregator.

        Args:
        X_train (np.ndarray): The training input data.
        y_train (np.ndarray): The training output data.
        X_val (np.ndarray): The validation input data.
        y_val (np.ndarray): The validation output data.
        epochs (int): The number of epochs to train for.
        batch_size (int): The batch size to use for training.
        learning_rate (float): The learning rate to use for training.
        """
        try:
            # Convert data to tensors
            X_train_tensor = torch.from_numpy(X_train).float().to(self.device)
            y_train_tensor = torch.from_numpy(y_train).float().to(self.device)
            X_val_tensor = torch.from_numpy(X_val).float().to(self.device)
            y_val_tensor = torch.from_numpy(y_val).float().to(self.device)

            # Define the loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

            # Train the model
            for epoch in range(epochs):
                # Train on batches
                for i in range(0, len(X_train_tensor), batch_size):
                    # Get the batch
                    batch_X = X_train_tensor[i:i+batch_size]
                    batch_y = y_train_tensor[i:i+batch_size]

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(batch_X)

                    # Compute the loss
                    loss = criterion(outputs, batch_y)

                    # Backward pass
                    loss.backward()

                    # Update the model parameters
                    optimizer.step()

                # Evaluate on validation set
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    logger.info(f'Epoch {epoch+1}, Val Loss: {val_loss.item()}')

        except Exception as e:
            logger.error(f'Training failed: {str(e)}')

    def predict_with_aggregator(self, X_test: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained MLPAggregator.

        Args:
        X_test (np.ndarray): The input data to make predictions on.

        Returns:
        np.ndarray: The predicted output data.
        """
        try:
            # Convert data to tensor
            X_test_tensor = torch.from_numpy(X_test).float().to(self.device)

            # Make predictions
            with torch.no_grad():
                outputs = self.model(X_test_tensor)

            # Convert outputs to numpy array
            predictions = outputs.cpu().numpy()

            return predictions

        except Exception as e:
            logger.error(f'Prediction failed: {str(e)}')
            return None

    def compute_combined_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the combined loss (MSE + NMSE).

        Args:
        y_pred (np.ndarray): The predicted output data.
        y_true (np.ndarray): The true output data.

        Returns:
        float: The combined loss.
        """
        try:
            # Compute MSE
            mse = mean_squared_error(y_true, y_pred)

            # Compute NMSE
            nmse = mse / np.var(y_true)

            # Compute combined loss
            combined_loss = mse + nmse

            return combined_loss

        except Exception as e:
            logger.error(f'Combined loss computation failed: {str(e)}')
            return None

    def extract_rollup_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extracts rollup features from the input data.

        Args:
        X (np.ndarray): The input data.

        Returns:
        np.ndarray: The extracted rollup features.
        """
        try:
            # Extract rollup features (e.g., mean, std, etc.)
            rollup_features = np.array([np.mean(X, axis=0), np.std(X, axis=0)])

            return rollup_features

        except Exception as e:
            logger.error(f'Rollup feature extraction failed: {str(e)}')
            return None

def main():
    # Example usage
    input_dim = 10
    hidden_dim = 20
    output_dim = 1
    device = 'cpu'

    neural_aggregator = NeuralAggregator(input_dim, hidden_dim, output_dim, device)

    # Generate some example data
    X_train = np.random.rand(100, input_dim)
    y_train = np.random.rand(100, output_dim)
    X_val = np.random.rand(20, input_dim)
    y_val = np.random.rand(20, output_dim)
    X_test = np.random.rand(10, input_dim)

    # Train the aggregator
    neural_aggregator.train_aggregator(X_train, y_train, X_val, y_val)

    # Make predictions
    predictions = neural_aggregator.predict_with_aggregator(X_test)

    # Compute combined loss
    combined_loss = neural_aggregator.compute_combined_loss(predictions, y_val)

    # Extract rollup features
    rollup_features = neural_aggregator.extract_rollup_features(X_train)

    logger.info(f'Predictions: {predictions}')
    logger.info(f'Combined Loss: {combined_loss}')
    logger.info(f'Rollup Features: {rollup_features}')

if __name__ == '__main__':
    main()