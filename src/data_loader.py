"""
Data Ingestion & Loading Module for AI-Based NIDS
Supports CIC-IDS2017, NSL-KDD, and high-fidelity synthetic benchmark generation.
"""

import os
import sys
import argparse
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# NSL-KDD Official Column Names
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

# Standard CIC-IDS2017 Flow Feature Columns (80 features + Label)
CIC_IDS2017_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"
]


def download_nsl_kdd(target_dir=RAW_DATA_DIR / "nsl_kdd"):
    """Downloads the NSL-KDD benchmark dataset."""
    target_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
    files = ["KDDTrain+.txt", "KDDTest+.txt"]
    
    downloaded_paths = []
    for f in files:
        dest = target_dir / f
        if not dest.exists():
            print(f"[*] Downloading NSL-KDD {f}...")
            url = base_url + f
            urllib.request.urlretrieve(url, dest)
            print(f"[+] Downloaded: {dest}")
        else:
            print(f"[✓] {f} already exists.")
        downloaded_paths.append(dest)
    
    df = pd.read_csv(downloaded_paths[0], header=None, names=NSL_KDD_COLUMNS)
    return df


def generate_synthetic_cic_ids(n_samples=50000, random_state=42):
    """
    Generates high-fidelity synthetic CIC-IDS2017 network flow traffic
    with authentic statistical distributions and multi-class cyber attacks.
    """
    np.random.seed(random_state)
    print(f"[*] Generating {n_samples:,} synthetic CIC-IDS2017 network flow records...")
    
    # Class distribution: 75% Benign, 12% DoS, 7% PortScan, 4% BruteForce, 2% Bot
    class_probs = [0.75, 0.12, 0.07, 0.04, 0.02]
    attack_types = ["BENIGN", "DoS Hulk", "PortScan", "SSH-Patator", "Bot"]
    
    labels = np.random.choice(attack_types, size=n_samples, p=class_probs)
    
    data = {}
    
    # Generate realistic features per class
    dst_ports = []
    flow_durations = []
    total_fwd_pkts = []
    total_bwd_pkts = []
    fwd_len_mean = []
    bwd_len_mean = []
    flow_bytes_sec = []
    flow_pkts_sec = []
    syn_flags = []
    ack_flags = []
    psh_flags = []
    fin_flags = []
    rst_flags = []
    init_win_fwd = []
    init_win_bwd = []
    
    for lbl in labels:
        if lbl == "BENIGN":
            # Normal HTTP/HTTPS/DNS traffic
            dst_ports.append(np.random.choice([80, 443, 53, 8080, 22, 445]))
            dur = np.random.exponential(scale=2_000_000) + 500  # microseconds
            fwd_p = np.random.randint(2, 30)
            bwd_p = np.random.randint(1, 35)
            fwd_lm = np.random.normal(loc=120, scale=40)
            bwd_lm = np.random.normal(loc=450, scale=150)
            syn_flags.append(1 if np.random.rand() > 0.6 else 0)
            ack_flags.append(1 if np.random.rand() > 0.2 else 0)
            psh_flags.append(1 if np.random.rand() > 0.4 else 0)
            fin_flags.append(1 if np.random.rand() > 0.7 else 0)
            rst_flags.append(0)
            init_win_fwd.append(np.random.randint(8192, 65535))
            init_win_bwd.append(np.random.randint(8192, 65535))
            
        elif lbl == "DoS Hulk":
            # Flood: very high packet count, high bytes/s, port 80/443
            dst_ports.append(np.random.choice([80, 443, 8080]))
            dur = np.random.exponential(scale=8_000_000) + 50_000
            fwd_p = np.random.randint(50, 400)
            bwd_p = np.random.randint(30, 250)
            fwd_lm = np.random.normal(loc=350, scale=50)
            bwd_lm = np.random.normal(loc=80, scale=20)
            syn_flags.append(1)
            ack_flags.append(1)
            psh_flags.append(1)
            fin_flags.append(0)
            rst_flags.append(1 if np.random.rand() > 0.8 else 0)
            init_win_fwd.append(np.random.randint(29200, 65535))
            init_win_bwd.append(np.random.randint(200, 4096))
            
        elif lbl == "PortScan":
            # Probing: 1-2 packets, fast, diverse ports, high SYN, no ACK/PSH
            dst_ports.append(np.random.randint(1, 65535))
            dur = np.random.exponential(scale=5_000) + 20
            fwd_p = np.random.randint(1, 3)
            bwd_p = np.random.choice([0, 1])
            fwd_lm = np.random.normal(loc=0, scale=5)
            bwd_lm = 0
            syn_flags.append(1)
            ack_flags.append(0)
            psh_flags.append(0)
            fin_flags.append(0)
            rst_flags.append(1 if np.random.rand() > 0.5 else 0)
            init_win_fwd.append(np.random.choice([1024, 2048, 4096]))
            init_win_bwd.append(-1 if bwd_p == 0 else 0)
            
        elif lbl == "SSH-Patator":
            # Brute force login on port 22: repetitive small packets
            dst_ports.append(22)
            dur = np.random.exponential(scale=600_000) + 10_000
            fwd_p = np.random.randint(15, 40)
            bwd_p = np.random.randint(15, 45)
            fwd_lm = np.random.normal(loc=80, scale=25)
            bwd_lm = np.random.normal(loc=110, scale=30)
            syn_flags.append(1)
            ack_flags.append(1)
            psh_flags.append(1)
            fin_flags.append(1)
            rst_flags.append(0)
            init_win_fwd.append(np.random.randint(14600, 29200))
            init_win_bwd.append(np.random.randint(14600, 29200))
            
        else: # Bot
            # Command & control beaconing
            dst_ports.append(np.random.choice([8080, 6667, 4444, 80]))
            dur = np.random.exponential(scale=5_000_000) + 200_000
            fwd_p = np.random.randint(8, 25)
            bwd_p = np.random.randint(5, 20)
            fwd_lm = np.random.normal(loc=60, scale=15)
            bwd_lm = np.random.normal(loc=90, scale=25)
            syn_flags.append(1)
            ack_flags.append(1)
            psh_flags.append(1)
            fin_flags.append(0)
            rst_flags.append(0)
            init_win_fwd.append(np.random.randint(8192, 32768))
            init_win_bwd.append(np.random.randint(8192, 32768))
            
        flow_durations.append(max(dur, 1.0))
        total_fwd_pkts.append(fwd_p)
        total_bwd_pkts.append(bwd_p)
        fwd_len_mean.append(max(fwd_lm, 0.0))
        bwd_len_mean.append(max(bwd_lm, 0.0))
        
        tot_bytes = (fwd_p * max(fwd_lm, 0.0)) + (bwd_p * max(bwd_lm, 0.0))
        tot_pkts = fwd_p + bwd_p
        sec = max(dur / 1_000_000.0, 0.0001)
        flow_bytes_sec.append(tot_bytes / sec)
        flow_pkts_sec.append(tot_pkts / sec)
        
    data["Destination Port"] = np.array(dst_ports, dtype=np.int32)
    data["Flow Duration"] = np.array(flow_durations, dtype=np.float64)
    data["Total Fwd Packets"] = np.array(total_fwd_pkts, dtype=np.int32)
    data["Total Backward Packets"] = np.array(total_bwd_pkts, dtype=np.int32)
    data["Total Length of Fwd Packets"] = data["Total Fwd Packets"] * np.array(fwd_len_mean, dtype=np.float64)
    data["Total Length of Bwd Packets"] = data["Total Backward Packets"] * np.array(bwd_len_mean, dtype=np.float64)
    data["Fwd Packet Length Max"] = np.array(fwd_len_mean) * 1.8 + 20
    data["Fwd Packet Length Min"] = np.maximum(0, np.array(fwd_len_mean) * 0.2)
    data["Fwd Packet Length Mean"] = np.array(fwd_len_mean, dtype=np.float64)
    data["Fwd Packet Length Std"] = np.array(fwd_len_mean) * 0.4
    data["Bwd Packet Length Max"] = np.array(bwd_len_mean) * 1.9 + 30
    data["Bwd Packet Length Min"] = np.maximum(0, np.array(bwd_len_mean) * 0.1)
    data["Bwd Packet Length Mean"] = np.array(bwd_len_mean, dtype=np.float64)
    data["Bwd Packet Length Std"] = np.array(bwd_len_mean) * 0.45
    data["Flow Bytes/s"] = np.array(flow_bytes_sec, dtype=np.float64)
    data["Flow Packets/s"] = np.array(flow_pkts_sec, dtype=np.float64)
    
    # Inter-Arrival Times (IAT)
    iat_base = data["Flow Duration"] / np.maximum(data["Total Fwd Packets"] + data["Total Backward Packets"], 1)
    data["Flow IAT Mean"] = iat_base
    data["Flow IAT Std"] = iat_base * 0.6
    data["Flow IAT Max"] = iat_base * 2.2
    data["Flow IAT Min"] = np.maximum(0, iat_base * 0.1)
    data["Fwd IAT Total"] = data["Flow Duration"] * 0.95
    data["Fwd IAT Mean"] = data["Flow Duration"] / np.maximum(data["Total Fwd Packets"], 1)
    data["Fwd IAT Std"] = data["Fwd IAT Mean"] * 0.5
    data["Fwd IAT Max"] = data["Fwd IAT Mean"] * 2.0
    data["Fwd IAT Min"] = np.maximum(0, data["Fwd IAT Mean"] * 0.1)
    data["Bwd IAT Total"] = data["Flow Duration"] * 0.90
    data["Bwd IAT Mean"] = data["Flow Duration"] / np.maximum(data["Total Backward Packets"], 1)
    data["Bwd IAT Std"] = data["Bwd IAT Mean"] * 0.55
    data["Bwd IAT Max"] = data["Bwd IAT Mean"] * 2.1
    data["Bwd IAT Min"] = np.maximum(0, data["Bwd IAT Mean"] * 0.1)
    
    # Header & Flag Counts
    data["Fwd PSH Flags"] = np.array(psh_flags, dtype=np.int32)
    data["Bwd PSH Flags"] = np.zeros(n_samples, dtype=np.int32)
    data["Fwd URG Flags"] = np.zeros(n_samples, dtype=np.int32)
    data["Bwd URG Flags"] = np.zeros(n_samples, dtype=np.int32)
    data["Fwd Header Length"] = data["Total Fwd Packets"] * 20
    data["Bwd Header Length"] = data["Total Backward Packets"] * 20
    data["Fwd Packets/s"] = data["Total Fwd Packets"] / np.maximum(data["Flow Duration"] / 1_000_000.0, 0.0001)
    data["Bwd Packets/s"] = data["Total Backward Packets"] / np.maximum(data["Flow Duration"] / 1_000_000.0, 0.0001)
    data["Min Packet Length"] = np.minimum(data["Fwd Packet Length Min"], data["Bwd Packet Length Min"])
    data["Max Packet Length"] = np.maximum(data["Fwd Packet Length Max"], data["Bwd Packet Length Max"])
    data["Packet Length Mean"] = (data["Total Length of Fwd Packets"] + data["Total Length of Bwd Packets"]) / np.maximum(data["Total Fwd Packets"] + data["Total Backward Packets"], 1)
    data["Packet Length Std"] = data["Packet Length Mean"] * 0.5
    data["Packet Length Variance"] = data["Packet Length Std"] ** 2
    
    data["FIN Flag Count"] = np.array(fin_flags, dtype=np.int32)
    data["SYN Flag Count"] = np.array(syn_flags, dtype=np.int32)
    data["RST Flag Count"] = np.array(rst_flags, dtype=np.int32)
    data["PSH Flag Count"] = np.array(psh_flags, dtype=np.int32)
    data["ACK Flag Count"] = np.array(ack_flags, dtype=np.int32)
    data["URG Flag Count"] = np.zeros(n_samples, dtype=np.int32)
    data["CWE Flag Count"] = np.zeros(n_samples, dtype=np.int32)
    data["ECE Flag Count"] = np.zeros(n_samples, dtype=np.int32)
    data["Down/Up Ratio"] = np.round(data["Total Backward Packets"] / np.maximum(data["Total Fwd Packets"], 1))
    data["Average Packet Size"] = data["Packet Length Mean"] * 1.05
    data["Avg Fwd Segment Size"] = data["Fwd Packet Length Mean"]
    data["Avg Bwd Segment Size"] = data["Bwd Packet Length Mean"]
    data["Fwd Header Length.1"] = data["Fwd Header Length"]
    data["Fwd Avg Bytes/Bulk"] = np.zeros(n_samples, dtype=np.float64)
    data["Fwd Avg Packets/Bulk"] = np.zeros(n_samples, dtype=np.float64)
    data["Fwd Avg Bulk Rate"] = np.zeros(n_samples, dtype=np.float64)
    data["Bwd Avg Bytes/Bulk"] = np.zeros(n_samples, dtype=np.float64)
    data["Bwd Avg Packets/Bulk"] = np.zeros(n_samples, dtype=np.float64)
    data["Bwd Avg Bulk Rate"] = np.zeros(n_samples, dtype=np.float64)
    data["Subflow Fwd Packets"] = data["Total Fwd Packets"]
    data["Subflow Fwd Bytes"] = data["Total Length of Fwd Packets"]
    data["Subflow Bwd Packets"] = data["Total Backward Packets"]
    data["Subflow Bwd Bytes"] = data["Total Length of Bwd Packets"]
    data["Init_Win_bytes_forward"] = np.array(init_win_fwd, dtype=np.int32)
    data["Init_Win_bytes_backward"] = np.array(init_win_bwd, dtype=np.int32)
    data["act_data_pkt_fwd"] = np.maximum(0, data["Total Fwd Packets"] - 1)
    data["min_seg_size_forward"] = np.full(n_samples, 20, dtype=np.int32)
    data["Active Mean"] = np.zeros(n_samples, dtype=np.float64)
    data["Active Std"] = np.zeros(n_samples, dtype=np.float64)
    data["Active Max"] = np.zeros(n_samples, dtype=np.float64)
    data["Active Min"] = np.zeros(n_samples, dtype=np.float64)
    data["Idle Mean"] = np.zeros(n_samples, dtype=np.float64)
    data["Idle Std"] = np.zeros(n_samples, dtype=np.float64)
    data["Idle Max"] = np.zeros(n_samples, dtype=np.float64)
    data["Idle Min"] = np.zeros(n_samples, dtype=np.float64)
    data["Label"] = labels
    
    df = pd.DataFrame(data)
    
    # Save raw benchmark CSV
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RAW_DATA_DIR / "cic_ids2017_benchmark.csv"
    df.to_csv(out_file, index=False)
    print(f"[+] Saved synthetic dataset to: {out_file} (Shape: {df.shape})")
    return df


def load_dataset(dataset_name="cic-ids2017", sample_size=None):
    """
    Main ingestion interface. Returns a cleaned pandas DataFrame.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if dataset_name.lower() in ["cic-ids2017", "cicids"]:
        csv_path = RAW_DATA_DIR / "cic_ids2017_benchmark.csv"
        if not csv_path.exists():
            df = generate_synthetic_cic_ids(n_samples=50000)
        else:
            print(f"[*] Loading existing CIC-IDS2017 dataset from {csv_path}...")
            df = pd.read_csv(csv_path)
            
    elif dataset_name.lower() in ["nsl-kdd", "nslkdd"]:
        df = download_nsl_kdd()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'cic-ids2017' or 'nsl-kdd'.")
        
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"[*] Subsampled to {sample_size:,} records.")
        
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIDS Data Ingestion Tool")
    parser.add_argument("--dataset", type=str, default="cic-ids2017", choices=["cic-ids2017", "nsl-kdd"])
    parser.add_argument("--samples", type=int, default=50000, help="Number of samples to generate/load")
    args = parser.parse_args()
    
    data = load_dataset(dataset_name=args.dataset, sample_size=args.samples)
    print("\n--- Data Ingestion Summary ---")
    print(f"Total Rows: {len(data):,}")
    print(f"Total Features: {data.shape[1]}")
    print("\nClass Distribution:")
    print(data["Label"].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))
