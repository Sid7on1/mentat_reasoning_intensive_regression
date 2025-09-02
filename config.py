import dataclasses
import typing
import pathlib
import logging
import json
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class DataConfig:
    """
    Configuration for data settings.

    Attributes:
        data_path (str): Path to the data directory.
        train_file (str): Name of the training file.
        test_file (str): Name of the testing file.
        batch_size (int): Batch size for training.
        num_workers (int): Number of workers for data loading.
    """
    data_path: str
    train_file: str
    test_file: str
    batch_size: int
    num_workers: int

    def __post_init__(self):
        """
        Validate data configuration.
        """
        if not pathlib.Path(self.data_path).exists():
            raise ValueError(f"Data path {self.data_path} does not exist")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")
        if not isinstance(self.num_workers, int) or self.num_workers <= 0:
            raise ValueError("Number of workers must be a positive integer")

@dataclasses.dataclass
class ModelConfig:
    """
    Configuration for model settings.

    Attributes:
        model_name (str): Name of the model.
        num_layers (int): Number of layers in the model.
        hidden_size (int): Hidden size of the model.
        dropout (float): Dropout rate of the model.
    """
    model_name: str
    num_layers: int
    hidden_size: int
    dropout: float

    def __post_init__(self):
        """
        Validate model configuration.
        """
        if not isinstance(self.num_layers, int) or self.num_layers <= 0:
            raise ValueError("Number of layers must be a positive integer")
        if not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
            raise ValueError("Hidden size must be a positive integer")
        if not isinstance(self.dropout, float) or self.dropout < 0 or self.dropout > 1:
            raise ValueError("Dropout rate must be a float between 0 and 1")

@dataclasses.dataclass
class ExperimentConfig:
    """
    Configuration for experiment settings.

    Attributes:
        experiment_name (str): Name of the experiment.
        num_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for training.
        weight_decay (float): Weight decay for training.
    """
    experiment_name: str
    num_epochs: int
    learning_rate: float
    weight_decay: float

    def __post_init__(self):
        """
        Validate experiment configuration.
        """
        if not isinstance(self.num_epochs, int) or self.num_epochs <= 0:
            raise ValueError("Number of epochs must be a positive integer")
        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be a positive float")
        if not isinstance(self.weight_decay, float) or self.weight_decay < 0:
            raise ValueError("Weight decay must be a non-negative float")

@dataclasses.dataclass
class MentatConfig:
    """
    Configuration for MENTAT settings.

    Attributes:
        data_config (DataConfig): Data configuration.
        model_config (ModelConfig): Model configuration.
        experiment_config (ExperimentConfig): Experiment configuration.
    """
    data_config: DataConfig
    model_config: ModelConfig
    experiment_config: ExperimentConfig

    def __post_init__(self):
        """
        Validate MENTAT configuration.
        """
        if not isinstance(self.data_config, DataConfig):
            raise ValueError("Data configuration must be an instance of DataConfig")
        if not isinstance(self.model_config, ModelConfig):
            raise ValueError("Model configuration must be an instance of ModelConfig")
        if not isinstance(self.experiment_config, ExperimentConfig):
            raise ValueError("Experiment configuration must be an instance of ExperimentConfig")

def load_config(config_file: str) -> MentatConfig:
    """
    Load configuration from a JSON file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        MentatConfig: Loaded configuration.
    """
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file {config_file} not found")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse configuration file {config_file}: {e}")
        raise

    data_config = DataConfig(
        data_path=config_data['data_config']['data_path'],
        train_file=config_data['data_config']['train_file'],
        test_file=config_data['data_config']['test_file'],
        batch_size=config_data['data_config']['batch_size'],
        num_workers=config_data['data_config']['num_workers']
    )

    model_config = ModelConfig(
        model_name=config_data['model_config']['model_name'],
        num_layers=config_data['model_config']['num_layers'],
        hidden_size=config_data['model_config']['hidden_size'],
        dropout=config_data['model_config']['dropout']
    )

    experiment_config = ExperimentConfig(
        experiment_name=config_data['experiment_config']['experiment_name'],
        num_epochs=config_data['experiment_config']['num_epochs'],
        learning_rate=config_data['experiment_config']['learning_rate'],
        weight_decay=config_data['experiment_config']['weight_decay']
    )

    return MentatConfig(
        data_config=data_config,
        model_config=model_config,
        experiment_config=experiment_config
    )

def main():
    config_file = 'config.json'
    try:
        config = load_config(config_file)
        logger.info(f"Loaded configuration: {config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")

if __name__ == '__main__':
    main()