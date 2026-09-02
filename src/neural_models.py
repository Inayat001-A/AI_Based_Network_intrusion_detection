"""
Deep Learning Neural Network Architectures for AI-Based NIDS
Implements 1D-CNN and LSTM classifiers in PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class NIDS_CNN1D(nn.Module):
    """
    1D-Convolutional Neural Network for localized spatial/feature correlation detection in network telemetry.
    """
    def __init__(self, input_dim=30, num_classes=5, dropout_rate=0.3):
        super(NIDS_CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input shape: [batch_size, input_dim] -> [batch_size, 1, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        out = self.fc2(x)
        return out


class NIDS_LSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) network for sequence and temporal flow dynamics.
    """
    def __init__(self, input_dim=30, hidden_dim=64, num_layers=2, num_classes=5, dropout_rate=0.3):
        super(NIDS_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Input shape: [batch_size, input_dim] -> [batch_size, 1, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.fc(self.dropout(last_step))
        return out


def train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=128, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Compute class weights for loss function
    unique_classes, counts = np.unique(y_train, return_counts=True)
    weights = len(y_train) / (len(unique_classes) * counts)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
                
        train_loss = total_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
    return model, history
