"""
Unsupervised Deep Autoencoder for Zero-Day Network Intrusion & Anomaly Detection
Trained strictly on benign telemetry to flag novel anomalous threat distributions.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class AutoencoderNet(nn.Module):
    """
    Symmetric Deep Autoencoder Network for Flow Feature Reconstruction.
    """
    def __init__(self, input_dim=30, latent_dim=4):
        super(AutoencoderNet, self).__init__()
        
        # Encoder: 30 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, latent_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Decoder: 4 -> 8 -> 16 -> 30
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class ZeroDayAutoencoderDetector:
    """
    High-level Detector wrapping Autoencoder training, dynamic thresholding, and anomaly scoring.
    """
    def __init__(self, input_dim=30, latent_dim=4, threshold_percentile=99.0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.threshold_percentile = threshold_percentile
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoencoderNet(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        self.threshold = None
        self.fitted = False

    def fit(self, X_benign, epochs=20, batch_size=128, lr=0.001, val_split=0.15):
        """
        Trains exclusively on normal/benign network traffic.
        """
        print(f"[*] Training Unsupervised Autoencoder on {len(X_benign):,} Benign flows on {self.device}...")
        
        # Train/Val split of benign traffic
        val_size = int(len(X_benign) * val_split)
        train_x = X_benign[val_size:]
        val_x = X_benign[:val_size]
        
        train_dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                reconstructed = self.model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1:>2}/{epochs}] - Reconstruction MSE Loss: {total_loss/len(train_loader):.6f}")
                
        # Calculate dynamic threshold on validation benign flows
        val_errors = self.get_reconstruction_errors(val_x)
        self.threshold = float(np.percentile(val_errors, self.threshold_percentile))
        self.fitted = True
        print(f"[+] Dynamic Zero-Day Threshold ({self.threshold_percentile}th percentile): {self.threshold:.6f}")
        return self

    def get_reconstruction_errors(self, X):
        """Computes sample-wise MSE reconstruction error."""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        return errors

    def predict_anomalies(self, X):
        """
        Returns:
            is_anomaly (np.array): 1 if MSE > threshold (Zero-Day Anomaly), 0 if Normal.
            errors (np.array): Continuous reconstruction loss scores.
        """
        if not self.fitted:
            raise RuntimeError("Autoencoder must be fitted before scoring.")
        errors = self.get_reconstruction_errors(X)
        is_anomaly = (errors > self.threshold).astype(int)
        return is_anomaly, errors

    def save(self, models_dir=MODELS_DIR):
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        weights_path = models_dir / "nids_autoencoder.pt"
        config_path = models_dir / "autoencoder_config.json"
        
        torch.save(self.model.state_dict(), weights_path)
        with open(config_path, "w") as f:
            json.dump({
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "threshold_percentile": self.threshold_percentile,
                "threshold": self.threshold
            }, f, indent=4)
        print(f"[+] Saved Autoencoder model to: {weights_path}")
        print(f"[+] Saved Autoencoder config to: {config_path}")

    @classmethod
    def load(cls, models_dir=MODELS_DIR):
        models_dir = Path(models_dir)
        weights_path = models_dir / "nids_autoencoder.pt"
        config_path = models_dir / "autoencoder_config.json"
        
        with open(config_path, "r") as f:
            cfg = json.load(f)
            
        detector = cls(
            input_dim=cfg["input_dim"],
            latent_dim=cfg["latent_dim"],
            threshold_percentile=cfg["threshold_percentile"]
        )
        detector.model.load_state_dict(torch.load(weights_path, map_location=detector.device))
        detector.threshold = cfg["threshold"]
        detector.fitted = True
        return detector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Day Autoencoder Trainer")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    # Load processed training data
    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        print("[!] processed train.csv not found. Run preprocessor first.")
        sys.exit(1)
        
    df_train = pd.read_csv(train_path)
    # Target 0 corresponds to BENIGN
    X_benign = df_train[df_train["Target"] == 0].drop(columns=["Target"]).values
    
    detector = ZeroDayAutoencoderDetector(input_dim=X_benign.shape[1], latent_dim=4)
    detector.fit(X_benign, epochs=args.epochs)
    detector.save()
