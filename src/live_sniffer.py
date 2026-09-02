"""
Live Packet Sniffer & PCAP Stream Processing Engine
Captures wire packets, parses protocol layers, and streams completed flows to the inference engine.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Callable, Optional
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.flow_aggregator import FlowAggregator, NetworkFlow

try:
    from scapy.all import sniff, rdpcap, PcapReader, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


class LivePacketSniffer:
    """
    Dissects network packets from Live NICs, PCAP capture dumps, or simulated streams.
    """
    def __init__(self, flow_callback: Optional[Callable[[NetworkFlow], None]] = None, timeout_seconds: float = 15.0):
        self.flow_aggregator = FlowAggregator(timeout_seconds=timeout_seconds)
        self.flow_callback = flow_callback
        self.packet_count = 0

    def _handle_packet(self, scapy_pkt):
        self.packet_count += 1
        
        if not scapy_pkt.haslayer(IP):
            return
            
        ip_layer = scapy_pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        pkt_len = len(scapy_pkt)
        timestamp = float(scapy_pkt.time) if hasattr(scapy_pkt, "time") else time.time()
        
        protocol = "OTHER"
        src_port = 0
        dst_port = 0
        header_len = 20
        window_size = 0
        flags = {}
        
        if scapy_pkt.haslayer(TCP):
            protocol = "TCP"
            tcp_layer = scapy_pkt[TCP]
            src_port = tcp_layer.sport
            dst_port = tcp_layer.dport
            header_len = tcp_layer.dataofs * 4 if tcp_layer.dataofs else 20
            window_size = tcp_layer.window
            
            # Extract TCP Flags
            flag_str = str(tcp_layer.flags)
            flags = {
                "SYN": "S" in flag_str,
                "FIN": "F" in flag_str,
                "ACK": "A" in flag_str,
                "PSH": "P" in flag_str,
                "URG": "U" in flag_str,
                "RST": "R" in flag_str
            }
        elif scapy_pkt.haslayer(UDP):
            protocol = "UDP"
            udp_layer = scapy_pkt[UDP]
            src_port = udp_layer.sport
            dst_port = udp_layer.dport
            header_len = 8

        completed_flow = self.flow_aggregator.process_packet(
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            pkt_len=pkt_len,
            timestamp=timestamp,
            flags=flags,
            header_len=header_len,
            window_size=window_size
        )
        
        if completed_flow and self.flow_callback:
            self.flow_callback(completed_flow)
            
        # Periodic expiration check (every 50 packets)
        if self.packet_count % 50 == 0:
            expired = self.flow_aggregator.flush_expired_flows(current_time=timestamp)
            if self.flow_callback:
                for flow in expired:
                    self.flow_callback(flow)

    def sniff_live(self, interface: Optional[str] = None, count: int = 0, bpf_filter: str = "ip"):
        """Sniffs live wire packets from local network interface."""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy is required for live network sniffing.")
        print(f"[*] Starting Live Packet Capture on interface '{interface or 'Default'}' (filter: '{bpf_filter}')...")
        sniff(iface=interface, filter=bpf_filter, prn=self._handle_packet, count=count, store=0)
        
        # Flush remaining flows
        remaining = self.flow_aggregator.flush_all()
        if self.flow_callback:
            for flow in remaining:
                self.flow_callback(flow)

    def read_pcap(self, pcap_file: str):
        """Reads and dissects an offline .pcap / .pcapng file."""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy is required for PCAP parsing.")
        print(f"[*] Reading and processing PCAP dump: {pcap_file}...")
        with PcapReader(pcap_file) as pcap_reader:
            for pkt in pcap_reader:
                self._handle_packet(pkt)
                
        remaining = self.flow_aggregator.flush_all()
        if self.flow_callback:
            for flow in remaining:
                self.flow_callback(flow)

    def simulate_traffic_stream(self, test_csv: Path, max_flows: int = 100, delay_sec: float = 0.01):
        """
        Simulates live wire telemetry by streaming preprocessed or synthetic records.
        """
        print(f"[*] Streaming {max_flows} simulated network flows (delay: {delay_sec*1000:.1f}ms/flow)...")
        df = pd.read_csv(test_csv)
        selected_df = df.head(max_flows)
        
        for idx, row in selected_df.iterrows():
            # Build pseudo flow
            src_ip = f"192.168.1.{10 + (idx % 50)}"
            dst_ip = f"10.0.0.{1 + (idx % 10)}"
            src_port = 49152 + (idx % 10000)
            dst_port = 80 if idx % 2 == 0 else 443
            protocol = "TCP"
            
            flow = NetworkFlow(src_ip, dst_ip, src_port, dst_port, protocol, time.time())
            # Populate packet lengths from features
            fwd_len = int(row.get("Total Length of Fwd Packets", 500))
            flow.fwd_pkt_lens = [fwd_len] if fwd_len > 0 else [100]
            flow.all_pkt_lens = flow.fwd_pkt_lens.copy()
            flow.syn_count = int(row.get("SYN Flag Count", 0))
            flow.ack_count = int(row.get("ACK Flag Count", 1))
            
            if self.flow_callback:
                self.flow_callback(flow)
            time.sleep(delay_sec)
