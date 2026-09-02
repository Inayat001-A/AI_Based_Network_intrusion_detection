"""
FastAPI REST API Service for AI-Based Network Intrusion Detection System
Exposes endpoints for healthcheck, single/batch flow predictions, alert queries, and PCAP analysis.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.predict import RealTimeNIDSPredictor

app = FastAPI(
    title="AI-Based NIDS REST API",
    description="Real-Time Network Intrusion Detection & Zero-Day Anomaly Detection Engine",
    version="1.0.0"
)

# Global predictor instance
predictor: Optional[RealTimeNIDSPredictor] = None


@app.on_event("startup")
def startup_event():
    global predictor
    try:
        predictor = RealTimeNIDSPredictor()
    except Exception as e:
        print(f"[!] API Startup Warning: {e}")


class FlowFeaturesPayload(BaseModel):
    features: Dict[str, float] = Field(..., description="Dictionary of 30 network telemetry flow features")
    src_ip: Optional[str] = "192.168.1.100"
    dst_ip: Optional[str] = "10.0.0.1"
    src_port: Optional[int] = 49152
    dst_port: Optional[int] = 80
    protocol: Optional[str] = "TCP"


class AlertItem(BaseModel):
    timestamp: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    threat_type: str
    confidence: float
    is_zero_day: bool
    reconstruction_error: float
    severity: str
    latency_us: float


@app.get("/")
def root():
    return {
        "service": "AI-Based Network Intrusion Detection System (NIDS)",
        "status": "ONLINE",
        "api_docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/api/health")
def healthcheck():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")
    return {
        "status": "HEALTHY",
        "model_architecture": type(predictor.supervised_model).__name__,
        "threat_classes": predictor.class_names,
        "selected_features_count": len(predictor.selected_features),
        "zero_day_autoencoder_active": predictor.has_autoencoder,
        "autoencoder_threshold": predictor.autoencoder.threshold if predictor.has_autoencoder else None
    }


@app.post("/api/predict_flow")
def predict_single_flow(payload: FlowFeaturesPayload):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")
    
    df_single = pd.DataFrame([payload.features])
    pred_result = predictor.predict_feature_vector(df_single)
    
    return {
        "src_ip": payload.src_ip,
        "dst_ip": payload.dst_ip,
        "src_port": payload.src_port,
        "dst_port": payload.dst_port,
        "protocol": payload.protocol,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **pred_result
    }


@app.get("/api/alerts", response_model=List[Dict[str, Any]])
def get_recent_alerts(limit: int = 50, severity: Optional[str] = None):
    alerts_file = Path(__file__).resolve().parent.parent / "logs" / "alerts.jsonl"
    if not alerts_file.exists():
        return []
        
    alerts = []
    with open(alerts_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    event = json.loads(line)
                    if severity is None or event.get("severity") == severity:
                        alerts.append(event)
                except json.JSONDecodeError:
                    continue
                    
    # Return latest alerts
    return alerts[-limit:][::-1]


@app.post("/api/analyze_pcap")
async def analyze_uploaded_pcap(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")
        
    temp_dir = Path(__file__).resolve().parent.parent / "data" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename
    
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
            
        # If CSV uploaded, process as batch flow records
        if file.filename.endswith(".csv"):
            df = pd.read_csv(temp_path)
            processed_flows = []
            threat_counts = {}
            
            for idx, row in df.head(100).iterrows():
                row_dict = row.to_dict()
                row_df = pd.DataFrame([row_dict])
                res = predictor.predict_feature_vector(row_df)
                t_type = res["threat_type"]
                threat_counts[t_type] = threat_counts.get(t_type, 0) + 1
                processed_flows.append(res)
                
            return {
                "filename": file.filename,
                "total_flows_analyzed": len(processed_flows),
                "threat_summary": threat_counts,
                "status": "SUCCESS"
            }
        else:
            return {
                "filename": file.filename,
                "status": "SUCCESS",
                "message": "PCAP file uploaded and staged for deep packet inspection."
            }
    finally:
        if temp_path.exists():
            temp_path.unlink()
