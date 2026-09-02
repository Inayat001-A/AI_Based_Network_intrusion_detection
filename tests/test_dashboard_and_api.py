"""
Unit tests for FastAPI Backend Service and Dashboard UI Components.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.api import app, startup_event
from app.ui_components import (
    create_attack_gauge,
    create_threat_donut,
    create_port_target_chart,
    create_loss_histogram
)

# Initialize API test client
startup_event()
client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "threat_classes" in data
    assert data["selected_features_count"] == 30


def test_api_predict_flow():
    dummy_payload = {
        "features": {
            "Total Fwd Packets": 5.0,
            "Total Backward Packets": 2.0,
            "SYN Flag Count": 1.0,
            "Fwd Packet Length Mean": 250.0,
            "Bwd Packet Length Mean": 500.0,
            "Avg Fwd Segment Size": 250.0,
            "Avg Bwd Segment Size": 500.0
        },
        "src_ip": "192.168.1.150",
        "dst_ip": "10.0.0.1",
        "src_port": 54321,
        "dst_port": 80,
        "protocol": "TCP"
    }
    response = client.post("/api/predict_flow", json=dummy_payload)
    assert response.status_code == 200
    data = response.json()
    assert "threat_type" in data
    assert "confidence" in data
    assert "severity" in data
    assert "latency_us" in data


def test_api_get_alerts():
    response = client.get("/api/alerts?limit=10")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_ui_components_charts():
    # Test Gauge
    gauge = create_attack_gauge(25.5)
    assert gauge is not None
    
    # Test Donut
    donut = create_threat_donut({"BENIGN": 50, "DoS Hulk": 10, "PortScan": 5})
    assert donut is not None
    
    # Test Port Chart
    port_chart = create_port_target_chart({80: 20, 443: 35, 22: 5})
    assert port_chart is not None
    
    # Test Loss Histogram
    hist = create_loss_histogram(np.array([0.01, 0.02, 0.05, 0.12]), threshold=0.093256)
    assert hist is not None
