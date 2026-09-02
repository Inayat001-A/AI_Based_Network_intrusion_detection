"""
Unit tests for Bidirectional Flow Aggregation, Live Sniffing, and Real-Time Inference.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd
from src.flow_aggregator import FlowAggregator, NetworkFlow
from src.predict import RealTimeNIDSPredictor


def test_flow_aggregator_packet_processing():
    aggregator = FlowAggregator(timeout_seconds=5.0)
    
    # Send packet 1: Client -> Server (SYN)
    flow = aggregator.process_packet(
        src_ip="192.168.1.100",
        dst_ip="10.0.0.1",
        src_port=50000,
        dst_port=80,
        protocol="TCP",
        pkt_len=60,
        flags={"SYN": True, "ACK": False, "FIN": False},
        header_len=20,
        window_size=65535
    )
    assert flow is None  # Connection still active
    
    # Send packet 2: Server -> Client (SYN+ACK)
    flow = aggregator.process_packet(
        src_ip="10.0.0.1",
        dst_ip="192.168.1.100",
        src_port=80,
        dst_port=50000,
        protocol="TCP",
        pkt_len=60,
        flags={"SYN": True, "ACK": True, "FIN": False},
        header_len=20,
        window_size=65535
    )
    assert flow is None
    
    # Send packet 3: Client -> Server (FIN+ACK) -> Closes connection
    completed_flow = aggregator.process_packet(
        src_ip="192.168.1.100",
        dst_ip="10.0.0.1",
        src_port=50000,
        dst_port=80,
        protocol="TCP",
        pkt_len=120,
        flags={"FIN": True, "ACK": True},
        header_len=20
    )
    
    assert completed_flow is not None
    assert len(completed_flow.all_pkt_lens) == 3
    assert len(completed_flow.fwd_pkt_lens) == 2
    assert len(completed_flow.bwd_pkt_lens) == 1
    
    # Test feature extraction
    feat_dict = completed_flow.extract_feature_dict()
    assert len(feat_dict) == 30
    assert feat_dict["SYN Flag Count"] == 2
    assert feat_dict["FIN Flag Count"] == 1
    assert feat_dict["Total Fwd Packets"] == 2
    assert feat_dict["Total Backward Packets"] == 1


def test_realtime_predictor():
    predictor = RealTimeNIDSPredictor()
    
    # Create sample flow
    flow = NetworkFlow("192.168.1.50", "10.0.0.1", 49152, 80, "TCP", time.time())
    flow.fwd_pkt_lens = [500, 600, 700]
    flow.bwd_pkt_lens = [200, 300]
    flow.all_pkt_lens = [500, 600, 700, 200, 300]
    flow.ack_count = 5
    
    # Warmup and test
    _ = predictor.process_flow(flow)
    alert = predictor.process_flow(flow)
    
    assert alert is not None
    assert "threat_type" in alert
    assert "confidence" in alert
    assert "latency_us" in alert
    assert alert["latency_us"] < 50_000  # Latency under 50ms
    assert (predictor.alerts_jsonl).exists()
