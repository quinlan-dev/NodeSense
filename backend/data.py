"""
NodeSense data pipeline.

Two data sources feed the same training code:

1. Synthetic flows (default). Statistically realistic CICIDS-style flow
   features for benign traffic and five attack classes, so the full
   pipeline runs end to end without the 70GB CICIDS-2018 download.
   Each class is generated from distributions chosen to mirror how the
   attack actually looks at the flow level (see class notes below).

2. Real CICIDS-2018 CSVs via load_cicids(), which maps the dataset's
   column names onto the same 20-feature vector so a model trained on
   either source serves identically.

Flows are generated in sessions of SEQ_LEN consecutive flows so the
transformer sees temporal structure (e.g. a scan sweep is many near
identical tiny flows in a row, beaconing is periodic idle gaps).
"""

import numpy as np

SEQ_LEN = 16

FEATURE_NAMES = [
    "Flow Duration", "Total Fwd Packets", "Total Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Mean", "Bwd Packet Length Max",
    "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Fwd IAT Mean", "Bwd IAT Mean",
    "Fwd PSH Flags", "SYN Flag Count", "ACK Flag Count", "URG Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Idle Mean",
]

CLASS_NAMES = ["Benign", "DDoS", "Port Scan", "Brute Force", "Botnet", "Infiltration"]

# Column names as they appear in CICIDS-2018 CSVs, in FEATURE_NAMES order.
CICIDS_COLUMNS = [
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Mean", "Bwd Pkt Len Max",
    "Bwd Pkt Len Mean", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Fwd IAT Mean", "Bwd IAT Mean",
    "Fwd PSH Flags", "SYN Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg", "Idle Mean",
]

CICIDS_LABEL_MAP = {
    "Benign": "Benign",
    "DDOS attack-HOIC": "DDoS", "DDOS attack-LOIC-UDP": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "Bot": "Botnet",
    "FTP-BruteForce": "Brute Force", "SSH-Bruteforce": "Brute Force",
    "Brute Force -Web": "Brute Force", "Brute Force -XSS": "Brute Force",
    "Infilteration": "Infiltration",
}


def _lognorm(rng, mean, sigma, size):
    """Lognormal around a target mean, the natural shape for durations,
    byte counts, and inter-arrival times."""
    return rng.lognormal(np.log(max(mean, 1e-9)), sigma, size)


def _flows(rng, n, *, duration, fwd_pkts, bwd_pkts, fwd_len, bwd_len,
           iat_regularity, psh, syn, ack, urg, idle):
    """Assemble n flows from per-class scale parameters.

    Derived features (bytes/s, packets/s, down/up ratio, avg packet size)
    are computed from the sampled primitives rather than sampled
    independently, so feature correlations look like real traffic.
    """
    dur = _lognorm(rng, duration, 1.0, n)
    fp = np.maximum(1, _lognorm(rng, fwd_pkts, 0.6, n)).round()
    bp = np.maximum(0, _lognorm(rng, bwd_pkts, 0.8, n) - 0.3).round()
    fl_mean = _lognorm(rng, fwd_len, 0.4, n)
    bl_mean = _lognorm(rng, bwd_len, 0.5, n)
    fl_max = fl_mean * rng.uniform(1.0, 3.0, n)
    bl_max = bl_mean * rng.uniform(1.0, 3.0, n)

    total_pkts = fp + bp
    total_bytes = fp * fl_mean + bp * bl_mean
    dur_s = np.maximum(dur, 1e-4) / 1e6  # duration is in microseconds
    bytes_s = total_bytes / dur_s
    pkts_s = total_pkts / dur_s

    iat_mean = dur / np.maximum(total_pkts - 1, 1)
    # iat_regularity near 0 = machine-regular timing (scans, beacons),
    # near 1 = bursty human traffic
    iat_std = iat_mean * np.abs(rng.normal(iat_regularity, iat_regularity / 2 + 0.05, n))
    fwd_iat = dur / np.maximum(fp - 1, 1) * rng.uniform(0.8, 1.2, n)
    bwd_iat = dur / np.maximum(bp - 1, 1) * rng.uniform(0.8, 1.2, n)

    down_up = bp / np.maximum(fp, 1)
    avg_size = total_bytes / np.maximum(total_pkts, 1)

    return np.column_stack([
        dur, fp, bp, fl_max, fl_mean, bl_max, bl_mean, bytes_s, pkts_s,
        iat_mean, iat_std, fwd_iat, bwd_iat,
        rng.binomial(1, psh, n) * np.maximum(fp * 0.3, 1).round(),
        rng.binomial(1, syn, n),
        rng.binomial(1, ack, n) * np.maximum(total_pkts * 0.4, 1).round(),
        rng.binomial(1, urg, n),
        down_up, avg_size,
        _lognorm(rng, idle, 1.2, n),
    ]).astype(np.float32)


def _benign(rng, n):
    # Web browsing / normal service traffic: seconds-long flows, balanced
    # bidirectional exchange, server sends more than client (down/up > 1).
    return _flows(rng, n, duration=2e6, fwd_pkts=12, bwd_pkts=14,
                  fwd_len=220, bwd_len=800, iat_regularity=0.9,
                  psh=0.5, syn=0.3, ack=0.95, urg=0.01, idle=5e4)


def _ddos(rng, n):
    # Flood: extreme packet/byte rates, tiny inter-arrival times, SYN-heavy,
    # almost nothing coming back from the victim.
    return _flows(rng, n, duration=8e4, fwd_pkts=90, bwd_pkts=1,
                  fwd_len=60, bwd_len=40, iat_regularity=0.15,
                  psh=0.05, syn=0.9, ack=0.15, urg=0.02, idle=1e2)


def _portscan(rng, n):
    # Scan probe: one or two tiny SYN packets per flow, microsecond
    # durations, machine-regular timing, no payload exchange.
    return _flows(rng, n, duration=3e3, fwd_pkts=1.4, bwd_pkts=0.6,
                  fwd_len=44, bwd_len=40, iat_regularity=0.1,
                  psh=0.01, syn=0.98, ack=0.1, urg=0.0, idle=1e2)


def _bruteforce(rng, n):
    # Credential guessing: short repeated auth attempts, small payloads
    # both ways, PSH on every attempt, metronome-regular retry timing.
    return _flows(rng, n, duration=4e5, fwd_pkts=8, bwd_pkts=7,
                  fwd_len=90, bwd_len=120, iat_regularity=0.2,
                  psh=0.9, syn=0.7, ack=0.9, urg=0.0, idle=8e3)


def _botnet(rng, n):
    # C2 beaconing: long mostly-idle flows with small periodic check-ins,
    # very high Idle Mean, low timing variance.
    return _flows(rng, n, duration=3e7, fwd_pkts=6, bwd_pkts=5,
                  fwd_len=130, bwd_len=180, iat_regularity=0.12,
                  psh=0.6, syn=0.2, ack=0.9, urg=0.05, idle=4e6)


def _infiltration(rng, n):
    # Data exfiltration after compromise: long flows dominated by large
    # backward (server-to-attacker) transfers, down/up ratio far above benign.
    return _flows(rng, n, duration=1.5e7, fwd_pkts=10, bwd_pkts=60,
                  fwd_len=100, bwd_len=1200, iat_regularity=0.6,
                  psh=0.6, syn=0.3, ack=0.95, urg=0.03, idle=2e5)


_GENERATORS = [_benign, _ddos, _portscan, _bruteforce, _botnet, _infiltration]


def generate_sessions(n_sessions: int = 4000, seed: int = 42,
                      benign_frac: float = 0.55):
    """Generate labeled sessions of SEQ_LEN flows.

    Attack sessions interleave a few benign flows (real attacks share the
    wire with normal traffic), which forces the transformer to use the
    sequence, not just one flow.

    Returns:
        X: (n_sessions, SEQ_LEN, n_features) raw feature values
        y: (n_sessions,) session class index into CLASS_NAMES
        y_flow: (n_sessions, SEQ_LEN) per-flow class index, for the
                per-flow baselines (random forest, autoencoder)
    """
    rng = np.random.default_rng(seed)
    n_feat = len(FEATURE_NAMES)
    X = np.zeros((n_sessions, SEQ_LEN, n_feat), dtype=np.float32)
    y = np.zeros(n_sessions, dtype=np.int64)
    y_flow = np.zeros((n_sessions, SEQ_LEN), dtype=np.int64)

    n_benign = int(n_sessions * benign_frac)
    classes = np.concatenate([
        np.zeros(n_benign, dtype=np.int64),
        rng.integers(1, len(CLASS_NAMES), n_sessions - n_benign),
    ])
    rng.shuffle(classes)

    for i, cls in enumerate(classes):
        y[i] = cls
        if cls == 0:
            X[i] = _benign(rng, SEQ_LEN)
        else:
            n_mix = rng.integers(0, 5)  # benign flows mixed into the session
            attack_idx = rng.permutation(SEQ_LEN) >= n_mix
            X[i][attack_idx] = _GENERATORS[cls](rng, int(attack_idx.sum()))
            X[i][~attack_idx] = _benign(rng, int((~attack_idx).sum()))
            y_flow[i][attack_idx] = cls
    return X, y, y_flow


def load_cicids(paths, max_rows_per_file: int = 200_000):
    """Load real CICIDS-2018 CSVs into the same session format.

    Rows are kept in file order (the CSVs are time ordered) and chunked
    into SEQ_LEN sessions; a session takes the majority attack label of
    its flows, or Benign if none.
    """
    import pandas as pd

    frames = []
    for path in paths:
        df = pd.read_csv(path, nrows=max_rows_per_file, skipinitialspace=True)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=CICIDS_COLUMNS + ["Label"])
        label = df["Label"].map(CICIDS_LABEL_MAP)
        keep = label.notna()
        sub = df.loc[keep, CICIDS_COLUMNS].astype(np.float32)
        sub["__label"] = label[keep].map({c: i for i, c in enumerate(CLASS_NAMES)})
        frames.append(sub)
    data = pd.concat(frames, ignore_index=True)

    n = (len(data) // SEQ_LEN) * SEQ_LEN
    X = data.iloc[:n, :-1].values.reshape(-1, SEQ_LEN, len(FEATURE_NAMES))
    y_flow = data["__label"].values[:n].reshape(-1, SEQ_LEN).astype(np.int64)
    # majority attack class per session, benign only if all flows benign
    y = np.zeros(len(X), dtype=np.int64)
    for i, row in enumerate(y_flow):
        attacks = row[row > 0]
        if len(attacks):
            y[i] = np.bincount(attacks).argmax()
    return X.astype(np.float32), y, y_flow
