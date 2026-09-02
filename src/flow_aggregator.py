"""
Bidirectional Network Flow Aggregator & On-the-Fly Feature Extraction Engine
Reconstructs 5-tuple flows from raw packet streams and calculates CIC-IDS2017 flow metrics.
"""

import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple


class NetworkFlow:
    """
    Tracks state, directional packet series, and statistical metrics for a single 5-tuple flow.
    """
    def __init__(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: str, start_time: float):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        
        self.start_time = start_time
        self.last_seen = start_time
        
        # Packet length series
        self.fwd_pkt_lens: List[int] = []
        self.bwd_pkt_lens: List[int] = []
        self.all_pkt_lens: List[int] = []
        
        # Timestamps
        self.fwd_timestamps: List[float] = []
        self.bwd_timestamps: List[float] = []
        
        # Header lengths & Flags
        self.fwd_header_len = 0
        self.bwd_header_len = 0
        
        self.syn_count = 0
        self.fin_count = 0
        self.ack_count = 0
        self.psh_count = 0
        self.urg_count = 0
        
        self.init_win_bytes_fwd = 0
        self.init_win_bytes_bwd = 0
        self.act_data_pkt_fwd = 0

    def add_packet(self, pkt_len: int, timestamp: float, is_forward: bool, flags: Dict[str, bool], header_len: int = 20, window_size: int = 0):
        self.last_seen = timestamp
        self.all_pkt_lens.append(pkt_len)
        
        if is_forward:
            self.fwd_pkt_lens.append(pkt_len)
            self.fwd_timestamps.append(timestamp)
            self.fwd_header_len += header_len
            if self.init_win_bytes_fwd == 0 and window_size > 0:
                self.init_win_bytes_fwd = window_size
            if pkt_len > header_len:
                self.act_data_pkt_fwd += 1
        else:
            self.bwd_pkt_lens.append(pkt_len)
            self.bwd_timestamps.append(timestamp)
            self.bwd_header_len += header_len
            if self.init_win_bytes_bwd == 0 and window_size > 0:
                self.init_win_bytes_bwd = window_size

        # Flag accounting
        if flags.get("SYN", False):
            self.syn_count += 1
        if flags.get("FIN", False):
            self.fin_count += 1
        if flags.get("ACK", False):
            self.ack_count += 1
        if flags.get("PSH", False):
            self.psh_count += 1
        if flags.get("URG", False):
            self.urg_count += 1

    def extract_feature_dict(self) -> Dict[str, float]:
        """Calculates exact 30 selected features for AI inference."""
        tot_fwd = len(self.fwd_pkt_lens)
        tot_bwd = len(self.bwd_pkt_lens)
        tot_pkts = len(self.all_pkt_lens)
        
        fwd_len_sum = sum(self.fwd_pkt_lens)
        bwd_len_sum = sum(self.bwd_pkt_lens)
        
        # Forward lengths
        fwd_max = max(self.fwd_pkt_lens) if tot_fwd > 0 else 0.0
        fwd_min = min(self.fwd_pkt_lens) if tot_fwd > 0 else 0.0
        fwd_mean = float(np.mean(self.fwd_pkt_lens)) if tot_fwd > 0 else 0.0
        fwd_std = float(np.std(self.fwd_pkt_lens)) if tot_fwd > 0 else 0.0
        
        # Backward lengths
        bwd_max = max(self.bwd_pkt_lens) if tot_bwd > 0 else 0.0
        bwd_min = min(self.bwd_pkt_lens) if tot_bwd > 0 else 0.0
        bwd_mean = float(np.mean(self.bwd_pkt_lens)) if tot_bwd > 0 else 0.0
        bwd_std = float(np.std(self.bwd_pkt_lens)) if tot_bwd > 0 else 0.0
        
        # All packet stats
        pkt_max = max(self.all_pkt_lens) if tot_pkts > 0 else 0.0
        pkt_min = min(self.all_pkt_lens) if tot_pkts > 0 else 0.0
        pkt_mean = float(np.mean(self.all_pkt_lens)) if tot_pkts > 0 else 0.0
        pkt_std = float(np.std(self.all_pkt_lens)) if tot_pkts > 0 else 0.0
        pkt_var = float(np.var(self.all_pkt_lens)) if tot_pkts > 0 else 0.0
        
        avg_pkt_size = (fwd_len_sum + bwd_len_sum) / tot_pkts if tot_pkts > 0 else 0.0
        avg_fwd_seg_size = fwd_len_sum / tot_fwd if tot_fwd > 0 else 0.0
        avg_bwd_seg_size = bwd_len_sum / tot_bwd if tot_bwd > 0 else 0.0

        features = {
            "Avg Bwd Segment Size": avg_bwd_seg_size,
            "Bwd Packet Length Mean": bwd_mean,
            "Fwd Packet Length Std": fwd_std,
            "Bwd Packet Length Max": bwd_max,
            "Bwd Packet Length Std": bwd_std,
            "act_data_pkt_fwd": float(self.act_data_pkt_fwd),
            "Bwd Packet Length Min": bwd_min,
            "Total Backward Packets": float(tot_bwd),
            "Min Packet Length": pkt_min,
            "Init_Win_bytes_backward": float(self.init_win_bytes_bwd),
            "Fwd Packet Length Max": fwd_max,
            "Max Packet Length": pkt_max,
            "Fwd Header Length.1": float(self.fwd_header_len),
            "Total Length of Fwd Packets": float(fwd_len_sum),
            "Bwd Header Length": float(self.bwd_header_len),
            "Avg Fwd Segment Size": avg_fwd_seg_size,
            "SYN Flag Count": float(self.syn_count),
            "Total Fwd Packets": float(tot_fwd),
            "FIN Flag Count": float(self.fin_count),
            "Average Packet Size": avg_pkt_size,
            "Fwd Packet Length Min": fwd_min,
            "Packet Length Mean": pkt_mean,
            "Packet Length Std": pkt_std,
            "Init_Win_bytes_forward": float(self.init_win_bytes_fwd),
            "Fwd Packet Length Mean": fwd_mean,
            "Subflow Fwd Packets": float(tot_fwd),
            "ACK Flag Count": float(self.ack_count),
            "Fwd PSH Flags": float(self.psh_count),
            "Fwd Header Length": float(self.fwd_header_len),
            "Packet Length Variance": pkt_var
        }
        return features

    def to_feature_vector(self, feature_names: List[str]) -> np.ndarray:
        feat_dict = self.extract_feature_dict()
        return np.array([feat_dict.get(col, 0.0) for col in feature_names], dtype=np.float32)


class FlowAggregator:
    """
    Maintains a hash map of live network flows and triggers flow completions upon timeout / TCP FIN.
    """
    def __init__(self, timeout_seconds: float = 15.0):
        self.timeout_seconds = timeout_seconds
        self.active_flows: Dict[Tuple, NetworkFlow] = {}

    def _get_canonical_key(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: str) -> Tuple:
        # Canonical symmetric key across both directions
        if (src_ip, src_port) <= (dst_ip, dst_port):
            return (src_ip, dst_ip, src_port, dst_port, protocol)
        else:
            return (dst_ip, src_ip, dst_port, src_port, protocol)

    def process_packet(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str,
        pkt_len: int,
        timestamp: Optional[float] = None,
        flags: Optional[Dict[str, bool]] = None,
        header_len: int = 20,
        window_size: int = 0
    ) -> Optional[NetworkFlow]:
        """
        Ingests a dissected packet, updates flow state, and returns completed flow if FIN/RST received.
        """
        if timestamp is None:
            timestamp = time.time()
        if flags is None:
            flags = {}

        key = self._get_canonical_key(src_ip, dst_ip, src_port, dst_port, protocol)
        
        if key not in self.active_flows:
            # First packet defines originator (forward direction)
            self.active_flows[key] = NetworkFlow(src_ip, dst_ip, src_port, dst_port, protocol, timestamp)
            
        flow = self.active_flows[key]
        is_forward = (src_ip == flow.src_ip and src_port == flow.src_port)
        flow.add_packet(pkt_len, timestamp, is_forward, flags, header_len, window_size)
        
        # If TCP connection closed (FIN/RST), complete and return flow immediately
        if flags.get("FIN", False) or flags.get("RST", False):
            del self.active_flows[key]
            return flow
        return None

    def flush_expired_flows(self, current_time: Optional[float] = None) -> List[NetworkFlow]:
        """Emits all flows that have been inactive longer than timeout_seconds."""
        if current_time is None:
            current_time = time.time()
            
        expired_keys = []
        for key, flow in self.active_flows.items():
            if current_time - flow.last_seen > self.timeout_seconds:
                expired_keys.append(key)
                
        expired_flows = [self.active_flows.pop(k) for k in expired_keys]
        return expired_flows

    def flush_all(self) -> List[NetworkFlow]:
        """Emits all active flows."""
        flows = list(self.active_flows.values())
        self.active_flows.clear()
        return flows
