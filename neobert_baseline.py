import logging
import os
import sys
from typing import Dict, List, Tuple
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeoBERTRegressor(nn.Module):
    """
    A PyTorch module that implements the NeoBERT finetuning baseline for comparison with MENTAT.
    
    Attributes:
    bert_model (BertModel): The pre-trained BERT model.
    dropout (nn.Dropout): The dropout layer.
    regressor (nn.Linear): The linear regression layer.
    """
    def __init__(self, bert_model: BertModel, dropout: float = 0.1):
        """
        Initializes the NeoBERTRegressor module.
        
        Args:
        bert_model (BertModel): The pre-trained BERT model.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(NeoBERTRegressor, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the NeoBERTRegressor module.
        
        Args:
        input_ids (torch.Tensor): The input IDs.
        attention_mask (torch.Tensor): The attention mask.
        
        Returns:
        torch.Tensor: The output of the regressor.
        """
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.regressor(pooled_output)
        return outputs

class NeoBERTDataset(Dataset):
    """
    A PyTorch dataset class that loads and preprocesses the data for the NeoBERTRegressor.
    
    Attributes:
    data (pd.DataFrame): The data.
    tokenizer (BertTokenizer): The BERT tokenizer.
    max_length (int): The maximum length of the input sequence.
    """
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_length: int = 512):
        """
        Initializes the NeoBERTDataset class.
        
        Args:
        data (pd.DataFrame): The data.
        tokenizer (BertTokenizer): The BERT tokenizer.
        max_length (int, optional): The maximum length of the input sequence. Defaults to 512.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        
        Returns:
        int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a sample from the dataset.
        
        Args:
        idx (int): The index of the sample.
        
        Returns:
        Dict[str, torch.Tensor]: The sample.
        """
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float),
        }

def prepare_neobert_data(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[NeoBERTDataset, NeoBERTDataset]:
    """
    Prepares the data for the NeoBERTRegressor.
    
    Args:
    data (pd.DataFrame): The data.
    test_size (float, optional): The proportion of the data to use for testing. Defaults to 0.2.
    random_state (int, optional): The random state. Defaults to 42.
    
    Returns:
    Tuple[NeoBERTDataset, NeoBERTDataset]: The training and testing datasets.
    """
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = NeoBERTDataset(train_data, tokenizer)
    test_dataset = NeoBERTDataset(test_data, tokenizer)
    return train_dataset, test_dataset

def train_neobert(model: NeoBERTRegressor, device: torch.device, train_dataset: NeoBERTDataset, test_dataset: NeoBERTDataset, batch_size: int = 32, epochs: int = 5) -> NeoBERTRegressor:
    """
    Trains the NeoBERTRegressor model.
    
    Args:
    model (NeoBERTRegressor): The NeoBERTRegressor model.
    device (torch.device): The device to use for training.
    train_dataset (NeoBERTDataset): The training dataset.
    test_dataset (NeoBERTDataset): The testing dataset.
    batch_size (int, optional): The batch size. Defaults to 32.
    epochs (int, optional): The number of epochs. Defaults to 5.
    
    Returns:
    NeoBERTRegressor: The trained NeoBERTRegressor model.
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}')

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                total_loss += loss.item()

            logger.info(f'Test Loss: {total_loss / len(test_dataloader)}')

    return model

def evaluate_neobert(model: NeoBERTRegressor, device: torch.device, test_dataset: NeoBERTDataset, batch_size: int = 32) -> float:
    """
    Evaluates the NeoBERTRegressor model.
    
    Args:
    model (NeoBERTRegressor): The NeoBERTRegressor model.
    device (torch.device): The device to use for evaluation.
    test_dataset (NeoBERTDataset): The testing dataset.
    batch_size (int, optional): The batch size. Defaults to 32.
    
    Returns:
    float: The mean squared error of the model.
    """
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        total_loss = 0
        predictions = []
        labels = []
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.MSELoss()(outputs, batch_labels)

            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

        mse = mean_squared_error(labels, predictions)
        logger.info(f'MSE: {mse}')

    return mse

if __name__ == '__main__':
    # Load the data
    data = pd.read_csv('data.csv')

    # Prepare the data
    train_dataset, test_dataset = prepare_neobert_data(data)

    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeoBERTRegressor(BertModel.from_pretrained('bert-base-uncased'))
    model.to(device)

    # Train the model
    model = train_neobert(model, device, train_dataset, test_dataset)

    # Evaluate the model
    evaluate_neobert(model, device, test_dataset)