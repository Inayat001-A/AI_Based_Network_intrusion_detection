"""
Real-Time Inference Engine & Security Alerting System for AI-Based NIDS
Integrates Supervised Tree Ensembles with Unsupervised Autoencoders for Wire-Speed Detection.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessor import NIDSPreprocessor
# Ensure __main__ has NIDSPreprocessor for pickle unpickling compatibility
import __main__
setattr(__main__, "NIDSPreprocessor", NIDSPreprocessor)

from src.flow_aggregator import NetworkFlow
from src.autoencoder import ZeroDayAutoencoderDetector
from src.live_sniffer import LivePacketSniffer

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class RealTimeNIDSPredictor:
    """
    Sub-millisecond inference engine with Dual-Layer detection (Supervised + Unsupervised Zero-Day).
    """
    def __init__(self, models_dir: Path = MODELS_DIR, logs_dir: Path = LOGS_DIR):
        self.models_dir = Path(models_dir)
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.alerts_jsonl = self.logs_dir / "alerts.jsonl"
        self.alerts_csv = self.logs_dir / "alerts.csv"
        
        # 1. Load Preprocessor
        prep_path = self.models_dir / "preprocessor.joblib"
        if not prep_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {prep_path}")
        self.preprocessor = joblib.load(prep_path)
        
        # 2. Load Selected Feature Names & Class Mappings
        feat_meta_path = self.models_dir / "selected_features.json"
        with open(feat_meta_path, "r") as f:
            feat_meta = json.load(f)
        self.selected_features = feat_meta["selected_features"]
        self.class_names = feat_meta["classes"]
        
        # 3. Load Supervised Model (XGBoost)
        model_path = self.models_dir / "nids_best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        self.supervised_model = joblib.load(model_path)
        
        # 4. Load Unsupervised Autoencoder (Zero-Day Anomaly Detector)
        try:
            self.autoencoder = ZeroDayAutoencoderDetector.load(self.models_dir)
            self.has_autoencoder = True
        except Exception as e:
            print(f"[!] Warning: Autoencoder detector not loaded ({e}). Continuing with supervised model only.")
            self.has_autoencoder = False
            
        print("[+] Real-Time NIDS Inference Engine Initialized Successfully.")
        print(f"[+] Active Classifier: {type(self.supervised_model).__name__}")
        print(f"[+] Loaded Threat Classes ({len(self.class_names)}): {self.class_names}")

    def predict_feature_vector(self, raw_features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Scales features and runs dual-model inference.
        """
        start_inf = time.time()
        
        # Ensure all selected features exist in input DataFrame
        for col in self.selected_features:
            if col not in raw_features_df.columns:
                raw_features_df[col] = 0.0
                
        # Transform through fitted scaler
        X_scaled = self.preprocessor.scaler.transform(raw_features_df[self.selected_features])
        
        # 1. Supervised Multi-Class Prediction
        probs = self.supervised_model.predict_proba(X_scaled)[0]
        pred_class_idx = int(np.argmax(probs))
        confidence = float(probs[pred_class_idx])
        threat_type = self.class_names[pred_class_idx]
        
        # 2. Unsupervised Zero-Day Anomaly Check
        is_zero_day = False
        reconstruction_error = 0.0
        if self.has_autoencoder:
            is_anom_arr, err_arr = self.autoencoder.predict_anomalies(X_scaled)
            is_zero_day = bool(is_anom_arr[0] == 1)
            reconstruction_error = float(err_arr[0])
            
        latency_us = (time.time() - start_inf) * 1_000_000
        
        # Determine Severity Level
        if threat_type == "BENIGN" and not is_zero_day:
            severity = "INFO"
        elif is_zero_day:
            severity = "CRITICAL (ZERO-DAY)"
        elif threat_type in ["DoS Hulk", "SSH-Patator"]:
            severity = "HIGH"
        elif threat_type in ["PortScan", "Bot"]:
            severity = "MEDIUM"
        else:
            severity = "LOW"
            
        return {
            "threat_type": threat_type if not is_zero_day else f"Zero-Day Anomaly ({threat_type})",
            "confidence": round(confidence, 4),
            "is_zero_day": is_zero_day,
            "reconstruction_error": round(reconstruction_error, 6),
            "severity": severity,
            "latency_us": round(latency_us, 2)
        }

    def process_flow(self, flow: NetworkFlow) -> Dict[str, Any]:
        """
        Extracts telemetry from completed flow, infers threat, logs alert, and returns result.
        """
        feat_dict = flow.extract_feature_dict()
        df_flow = pd.DataFrame([feat_dict])
        
        pred_res = self.predict_feature_vector(df_flow)
        
        alert_event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip": flow.src_ip,
            "dst_ip": flow.dst_ip,
            "src_port": flow.src_port,
            "dst_port": flow.dst_port,
            "protocol": flow.protocol,
            "packets": len(flow.all_pkt_lens),
            "bytes": sum(flow.all_pkt_lens),
            **pred_res
        }
        
        # Log to JSONL
        with open(self.alerts_jsonl, "a") as f:
            f.write(json.dumps(alert_event) + "\n")
            
        # Log to CSV if attack
        if alert_event["threat_type"] != "BENIGN":
            alert_df = pd.DataFrame([alert_event])
            alert_df.to_csv(self.alerts_csv, mode="a", header=not self.alerts_csv.exists(), index=False)
            
            # Print Alert in Console
            print(f"[ALERT] [{alert_event['timestamp']}] [{alert_event['severity']}] "
                  f"{alert_event['src_ip']}:{alert_event['src_port']} -> {alert_event['dst_ip']}:{alert_event['dst_port']} | "
                  f"THREAT: {alert_event['threat_type']} (Conf: {alert_event['confidence']*100:.1f}%) | "
                  f"Latency: {alert_event['latency_us']:.1f}us")
        else:
            print(f"[OK]    [{alert_event['timestamp']}] BENIGN FLOW "
                  f"{alert_event['src_ip']}:{alert_event['src_port']} -> {alert_event['dst_ip']}:{alert_event['dst_port']} | "
                  f"Latency: {alert_event['latency_us']:.1f}us")
                  
        return alert_event


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time NIDS Predictor")
    parser.add_argument("--mode", type=str, default="stream", choices=["stream", "pcap", "live"])
    parser.add_argument("--pcap", type=str, default=None, help="Path to PCAP file")
    parser.add_argument("--samples", type=int, default=30, help="Number of flows to stream in test mode")
    parser.add_argument("--delay", type=float, default=0.01, help="Delay between flows in stream mode (seconds)")
    args = parser.parse_args()
    
    predictor = RealTimeNIDSPredictor()
    sniffer = LivePacketSniffer(flow_callback=predictor.process_flow)
    
    if args.mode == "stream":
        test_csv = PROCESSED_DIR / "test.csv"
        sniffer.simulate_traffic_stream(test_csv, max_flows=args.samples, delay_sec=args.delay)
    elif args.mode == "pcap":
        if not args.pcap or not Path(args.pcap).exists():
            print(f"[!] PCAP file not found: {args.pcap}")
            sys.exit(1)
        sniffer.read_pcap(args.pcap)
    elif args.mode == "live":
        sniffer.sniff_live()
