import logging
import json
import pickle
import hashlib
import datetime
import os
import random
import numpy as np
from typing import Dict, Any

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("utils.log"),
        logging.StreamHandler()
    ]
)

class Utils:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def setup_logging(self):
        """
        Set up logging configuration based on the provided configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing logging settings.

        Returns:
            None
        """
        level = config.get("logging_level", "INFO")
        self.logger.setLevel(getattr(logging, level.upper()))

        # Create file handler and set level to DEBUG
        file_handler = logging.FileHandler("utils.log")
        file_handler.setLevel(logging.DEBUG)

        # Create console handler and set level to INFO
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and attach it to the handlers
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def save_checkpoint(self, model, optimizer, epoch, metrics):
        """
        Save the model, optimizer, and metrics to a checkpoint file.

        Args:
            model (Any): Model to save.
            optimizer (Any): Optimizer to save.
            epoch (int): Current epoch number.
            metrics (Dict[str, float]): Metrics to save.

        Returns:
            None
        """
        checkpoint_path = os.path.join(self.config["checkpoint_dir"], f"checkpoint_{epoch}.pth")
        self.logger.info(f"Saving checkpoint to {checkpoint_path}")

        # Save model and optimizer
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics
        }

        # Save checkpoint to file
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint from a file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            Dict[str, Any]: Loaded checkpoint dictionary.
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        return checkpoint

    def set_random_seed(self, seed: int):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): Random seed to set.

        Returns:
            None
        """
        self.logger.info(f"Setting random seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)

    def log_experiment(self, experiment_name: str, metrics: Dict[str, float]):
        """
        Log an experiment with its metrics.

        Args:
            experiment_name (str): Name of the experiment.
            metrics (Dict[str, float]): Metrics to log.

        Returns:
            None
        """
        self.logger.info(f"Experiment: {experiment_name}")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value}")

def main():
    config = {
        "logging_level": "INFO",
        "checkpoint_dir": "checkpoints"
    }

    utils = Utils(config)
    utils.setup_logging()

    # Example usage
    model = None  # Replace with your model
    optimizer = None  # Replace with your optimizer
    epoch = 10
    metrics = {"accuracy": 0.9, "loss": 0.1}

    utils.save_checkpoint(model, optimizer, epoch, metrics)
    checkpoint = utils.load_checkpoint("checkpoints/checkpoint_10.pth")
    utils.set_random_seed(42)
    utils.log_experiment("Test Experiment", metrics)

if __name__ == "__main__":
    main()