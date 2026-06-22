"""
EMG Signal Quality Analyzer
A local Gradio app for scanning raw EMG files and extracting signal-quality information.

Expected input columns:
time ch1 ch2 ch3 ch4 ch5 ch6 ch7 ch8 label
"""

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

from pathlib import Path
import traceback

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_examples"
DEMO_FILE = DATA_DIR / "demo_emg_signal.txt"

CHANNELS = [f"ch{i}" for i in range(1, 9)]
EXPECTED_COLUMNS = ["time"] + CHANNELS + ["label"]
MAX_FILE_SIZE_MB = 50
MAX_ROWS = 1_000_000
DEFAULT_GAP_MS = 10.0
WINDOW_SIZE = 200
WINDOW_STEP = 100
VALID_LABELS = {0, 1, 2, 3, 4, 5, 6, 7}

LABEL_MAP = {
    0: "Unmarked / transition",
    1: "Rest / hold",
    2: "Fist / close hand",
    3: "Wrist flexion",
    4: "Wrist extension / open hand",
    5: "Radial deviation",
    6: "Ulnar deviation",
    7: "Relax / unclassified",
}

CSS = """
:root {
    --accent-red: #b91c1c;
    --accent-red-dark: #7f1d1d;
    --text-main: #111111;
    --text-muted: #4b5563;
    --border-light: #d1d5db;
    --panel-bg: #ffffff;
    --soft-red: #fff5f5;
}
body, .gradio-container {
    background: #ffffff !important;
    color: var(--text-main) !important;
    font-family: "Times New Roman", Times, serif !important;
    font-size: 17px !important;
    line-height: 1.55 !important;
}
.gradio-container * {
    font-family: "Times New Roman", Times, serif !important;
}
h1 {
    font-size: 32px !important;
    line-height: 1.2 !important;
    color: #111111 !important;
    font-weight: 700 !important;
}
h2, h3, h4 {
    color: #111111 !important;
    font-weight: 700 !important;
}
label, .label, .wrap, .prose, .markdown, textarea, input, select {
    font-size: 17px !important;
}
#title-block {
    border-left: 7px solid var(--accent-red);
    padding: 18px 22px;
    background: #ffffff;
    border-radius: 12px;
    border-top: 1px solid var(--border-light);
    border-right: 1px solid var(--border-light);
    border-bottom: 1px solid var(--border-light);
    box-shadow: 0 2px 10px rgba(17, 17, 17, 0.04);
}
#title-block h1 {
    margin-bottom: 8px !important;
}
#title-block p {
    color: var(--text-muted);
    font-size: 18px !important;
    margin: 4px 0 0 0;
}
button.primary, .primary button, button[variant="primary"] {
    background: var(--accent-red) !important;
    border-color: var(--accent-red-dark) !important;
    color: white !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
}
button.primary:hover, .primary button:hover {
    background: var(--accent-red-dark) !important;
}
button.secondary, .secondary button {
    border-color: var(--accent-red) !important;
}
.gr-box, .block, .form, .panel {
    border-radius: 12px !important;
}
.dataframe, table, th, td {
    font-size: 16px !important;
}
.metric-card {
    border: 1px solid var(--border-light);
    border-left: 5px solid var(--accent-red);
    border-radius: 12px;
    padding: 14px 16px;
    background: #ffffff;
    margin-bottom: 8px;
}
.metric-title {
    font-weight: 700;
    color: #111111;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: var(--accent-red);
}
.small-note {
    color: var(--text-muted);
    font-size: 15px;
}
.red-badge {
    display: inline-block;
    background: #fee2e2;
    color: #991b1b;
    border: 1px solid #fecaca;
    padding: 3px 10px;
    border-radius: 999px;
    font-weight: 700;
}
"""


def get_file_path(file_obj):
    """Return uploaded file path or demo file path."""
    if file_obj is None:
        return DEMO_FILE
    if isinstance(file_obj, str):
        return Path(file_obj)
    if hasattr(file_obj, "name"):
        return Path(file_obj.name)
    if isinstance(file_obj, dict):
        return Path(file_obj.get("path") or file_obj.get("name"))
    return DEMO_FILE


def validate_file_path(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix not in {".txt", ".csv"}:
        raise ValueError("Only .txt and .csv EMG files are allowed.")
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File is too large: {size_mb:.1f} MB. Limit is {MAX_FILE_SIZE_MB} MB.")


def read_emg_file(file_path: Path) -> pd.DataFrame:
    """Read raw EMG file robustly.

    Supported input:
    - .txt or .csv
    - with or without a header row
    - whitespace, comma or semicolon separators
    - first 10 columns interpreted as: time, ch1..ch8, label
    """
    validate_file_path(file_path)

    # Use one parser for txt/csv to avoid losing the first row in headerless files.
    # Header rows are handled by numeric conversion and then removed as non-numeric.
    try:
        df = pd.read_csv(
            file_path,
            sep=r"[\s,;]+",
            engine="python",
            header=None,
            nrows=MAX_ROWS,
            comment="#",
            on_bad_lines="skip",
        )
    except TypeError:
        # Compatibility with older pandas versions that do not support on_bad_lines.
        df = pd.read_csv(
            file_path,
            sep=r"[\s,;]+",
            engine="python",
            header=None,
            nrows=MAX_ROWS,
            comment="#",
        )

    if df.shape[1] < 10:
        raise ValueError(
            f"The file contains only {df.shape[1]} columns. Expected at least 10 columns: "
            "time ch1 ch2 ch3 ch4 ch5 ch6 ch7 ch8 label."
        )

    df = df.iloc[:, :10].copy()
    df.columns = EXPECTED_COLUMNS

    for col in EXPECTED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    initial_rows = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=EXPECTED_COLUMNS).copy()
    dropped_rows = initial_rows - len(df)

    if len(df) < WINDOW_SIZE:
        raise ValueError(
            f"Not enough valid rows after cleaning: {len(df)}. Need at least {WINDOW_SIZE} samples."
        )

    df = df.sort_values("time").reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df.attrs["dropped_rows"] = int(dropped_rows)
    df.attrs["initial_rows"] = int(initial_rows)
    return df

def estimate_sampling(df: pd.DataFrame):
    t = df["time"].to_numpy(dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt_pos = dt[dt > 0]
    if len(dt_pos) == 0:
        return {
            "median_dt": np.nan,
            "mean_dt": np.nan,
            "estimated_fs": np.nan,
            "gap_threshold": DEFAULT_GAP_MS,
            "gap_count": 0,
            "gap_percent": 0.0,
            "max_gap": np.nan,
            "irregular_percent": 0.0,
        }
    median_dt = float(np.median(dt_pos))
    mean_dt = float(np.mean(dt_pos))
    estimated_fs = 1000.0 / median_dt if median_dt > 0 else np.nan
    gap_threshold = max(DEFAULT_GAP_MS, 5.0 * median_dt)
    gap_mask = dt_pos > gap_threshold
    irregular_mask = np.abs(dt_pos - median_dt) > max(1.0, 0.25 * median_dt)
    return {
        "median_dt": median_dt,
        "mean_dt": mean_dt,
        "estimated_fs": estimated_fs,
        "gap_threshold": gap_threshold,
        "gap_count": int(gap_mask.sum()),
        "gap_percent": float(100.0 * gap_mask.mean()),
        "max_gap": float(np.max(dt_pos)),
        "irregular_percent": float(100.0 * irregular_mask.mean()),
    }


def robust_spike_rate(x: np.ndarray):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    robust_z = 0.6745 * np.abs(x - med) / mad
    return float(100.0 * np.mean(robust_z > 8.0))


def flatline_rate(x: np.ndarray):
    """Percentage of consecutive samples with almost no change."""
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 100.0
    diff = np.abs(np.diff(x))
    scale = np.percentile(np.abs(x), 95) + 1e-12
    threshold = max(1e-12, 0.0005 * scale)
    return float(100.0 * np.mean(diff <= threshold))


def compute_channel_metrics(df: pd.DataFrame):
    rows = []
    activity_values = []
    for ch in CHANNELS:
        x = df[ch].to_numpy(dtype=float)
        finite_x = x[np.isfinite(x)]
        if len(finite_x) == 0:
            metrics = dict(mean=np.nan, std=np.nan, rms=np.nan, mav=np.nan, p2_5=np.nan, p97_5=np.nan,
                           max_abs=np.nan, flatline_percent=100.0, spike_percent=0.0)
        else:
            metrics = {
                "mean": float(np.mean(finite_x)),
                "std": float(np.std(finite_x)),
                "rms": float(np.sqrt(np.mean(finite_x ** 2))),
                "mav": float(np.mean(np.abs(finite_x))),
                "p2_5": float(np.percentile(finite_x, 2.5)),
                "p97_5": float(np.percentile(finite_x, 97.5)),
                "max_abs": float(np.max(np.abs(finite_x))),
                "flatline_percent": flatline_rate(finite_x),
                "spike_percent": robust_spike_rate(finite_x),
            }
        activity_values.append(metrics["mav"] if np.isfinite(metrics["mav"]) else 0.0)
        rows.append({"Channel": ch, **metrics})

    activity_values = np.array(activity_values, dtype=float)
    denom = np.max(activity_values) if np.max(activity_values) > 0 else 1.0
    for row, activity in zip(rows, activity_values):
        row["Activity score"] = float(100.0 * activity / denom)
        row["Assessment"] = assess_channel(row)

    out = pd.DataFrame(rows)
    numeric_cols = [c for c in out.columns if c not in {"Channel", "Assessment"}]
    out[numeric_cols] = out[numeric_cols].round(6)
    return out


def assess_channel(row):
    flags = []
    if row.get("flatline_percent", 0) > 95:
        flags.append("possible flatline")
    if row.get("spike_percent", 0) > 1.0:
        flags.append("spikes")
    if row.get("Activity score", 0) < 10:
        flags.append("low activity")
    return "OK" if not flags else "; ".join(flags)


def compute_window_features(df: pd.DataFrame):
    X, y, times, purities = [], [], [], []
    signal = df[CHANNELS].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=int)
    times_arr = df["time"].to_numpy(dtype=float)

    for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_STEP):
        end = start + WINDOW_SIZE
        win = signal[start:end]
        win_labels = labels[start:end]
        counts = np.bincount(win_labels, minlength=8)
        main_label = int(np.argmax(counts))
        purity = float(np.mean(win_labels == main_label))
        features = []
        for i in range(len(CHANNELS)):
            ch = win[:, i]
            mav = float(np.mean(np.abs(ch)))
            rms = float(np.sqrt(np.mean(ch ** 2)))
            wl = float(np.sum(np.abs(np.diff(ch))))
            features.extend([mav, rms, wl])
        X.append(features)
        y.append(main_label)
        purities.append(purity)
        times.append(times_arr[start])

    return np.array(X), np.array(y), np.array(times), np.array(purities)


def compute_label_metrics(df: pd.DataFrame, X, y_win, purities):
    labels = sorted(pd.unique(df["label"]))
    total = len(df)
    rows = []
    overall_mav = df[CHANNELS].abs().mean(axis=1).to_numpy()
    for label in labels:
        sample_mask = df["label"].to_numpy() == label
        count = int(sample_mask.sum())
        valid_window_mask = (y_win == label) & (purities >= 0.90)
        rows.append({
            "Label": int(label),
            "Gesture name": LABEL_MAP.get(int(label), "Unknown"),
            "Samples": count,
            "Share, %": 100.0 * count / total if total > 0 else 0.0,
            "Valid 200 ms windows": int(valid_window_mask.sum()),
            "Mean absolute activity": float(np.mean(overall_mav[sample_mask])) if count > 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    out["Share, %"] = out["Share, %"].round(2)
    out["Mean absolute activity"] = out["Mean absolute activity"].round(6)
    return out


def detect_suspicious_intervals(df: pd.DataFrame, sampling_info, max_rows=50):
    rows = []
    t = df["time"].to_numpy(dtype=float)
    dt = np.diff(t)
    threshold = sampling_info.get("gap_threshold", DEFAULT_GAP_MS)
    gap_indices = np.where(dt > threshold)[0]
    for idx in gap_indices[:20]:
        rows.append({
            "Type": "time gap",
            "Start time": float(t[idx]),
            "End time": float(t[idx + 1]),
            "Channel": "all",
            "Value": float(dt[idx]),
            "Comment": f"Gap > {threshold:.2f} ms",
        })

    # Top robust spikes per channel. Limit to avoid excessive output.
    for ch in CHANNELS:
        x = df[ch].to_numpy(dtype=float)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med)) + 1e-12
        robust_z = 0.6745 * np.abs(x - med) / mad
        spike_idx = np.where(robust_z > 8.0)[0]
        if len(spike_idx) > 0:
            # keep strongest 3 spikes per channel
            strongest = spike_idx[np.argsort(robust_z[spike_idx])[-3:]][::-1]
            for idx in strongest:
                rows.append({
                    "Type": "amplitude spike",
                    "Start time": float(t[idx]),
                    "End time": float(t[idx]),
                    "Channel": ch,
                    "Value": float(x[idx]),
                    "Comment": f"robust z={robust_z[idx]:.1f}",
                })

    if not rows:
        rows.append({
            "Type": "none",
            "Start time": np.nan,
            "End time": np.nan,
            "Channel": "-",
            "Value": np.nan,
            "Comment": "No major suspicious intervals found by the current rules.",
        })
    return pd.DataFrame(rows).head(max_rows)


def compute_quality_score(df, sampling_info, channel_table, label_table):
    """Deterministic 0-100 score; higher is better."""
    score = 100.0
    reasons = []
    initial_rows = df.attrs.get("initial_rows", len(df))
    dropped_rows = df.attrs.get("dropped_rows", 0)
    dropped_rate = 100.0 * dropped_rows / max(initial_rows, 1)
    if dropped_rate > 0.5:
        penalty = min(15, dropped_rate * 2)
        score -= penalty
        reasons.append(f"Dropped rows after numeric cleaning: {dropped_rate:.2f}%")

    gap_percent = sampling_info.get("gap_percent", 0.0)
    if gap_percent > 0.5:
        penalty = min(20, gap_percent * 4)
        score -= penalty
        reasons.append(f"Time gaps: {gap_percent:.2f}%")

    irregular_percent = sampling_info.get("irregular_percent", 0.0)
    if irregular_percent > 5.0:
        penalty = min(10, (irregular_percent - 5.0) * 0.5)
        score -= penalty
        reasons.append(f"Irregular sampling intervals: {irregular_percent:.2f}%")

    spike_mean = float(channel_table["spike_percent"].mean()) if "spike_percent" in channel_table else 0.0
    if spike_mean > 0.5:
        penalty = min(20, spike_mean * 6)
        score -= penalty
        reasons.append(f"Amplitude spikes: average {spike_mean:.2f}%")

    flatline_channels = int((channel_table["flatline_percent"] > 95).sum()) if "flatline_percent" in channel_table else 0
    if flatline_channels > 0:
        penalty = min(25, flatline_channels * 6)
        score -= penalty
        reasons.append(f"Possible flatline channels: {flatline_channels}")

    labels_present = set(label_table["Label"].astype(int).tolist()) if len(label_table) else set()
    active_labels = labels_present.intersection({1, 2, 3, 4, 5, 6})
    if len(active_labels) < 6:
        penalty = (6 - len(active_labels)) * 4
        score -= penalty
        reasons.append(f"Only {len(active_labels)}/6 expected main gesture labels are present")

    score = max(0.0, min(100.0, score))
    if not reasons:
        reasons.append("No major quality problems detected by the current deterministic checks.")
    return round(score, 1), reasons


def build_overview(df, sampling_info, reasons, file_path):
    duration_ms = float(df["time"].iloc[-1] - df["time"].iloc[0]) if len(df) > 1 else np.nan
    duration_sec = duration_ms / 1000.0 if np.isfinite(duration_ms) else np.nan
    labels_present = sorted([int(x) for x in pd.unique(df["label"])])
    initial_rows = df.attrs.get("initial_rows", len(df))
    dropped_rows = df.attrs.get("dropped_rows", 0)
    rows = [
        {"Metric": "File", "Value": Path(file_path).name},
        {"Metric": "Valid samples", "Value": f"{len(df):,}"},
        {"Metric": "Rows dropped during cleaning", "Value": f"{dropped_rows:,} / {initial_rows:,}"},
        {"Metric": "Duration", "Value": f"{duration_sec:.2f} s" if np.isfinite(duration_sec) else "unknown"},
        {"Metric": "Estimated sampling rate", "Value": f"{sampling_info['estimated_fs']:.1f} Hz" if np.isfinite(sampling_info['estimated_fs']) else "unknown"},
        {"Metric": "Median dt", "Value": f"{sampling_info['median_dt']:.3f} ms" if np.isfinite(sampling_info['median_dt']) else "unknown"},
        {"Metric": "Time gaps", "Value": f"{sampling_info['gap_count']} gaps ({sampling_info['gap_percent']:.2f}%)"},
        {"Metric": "Max gap", "Value": f"{sampling_info['max_gap']:.2f} ms" if np.isfinite(sampling_info['max_gap']) else "unknown"},
        {"Metric": "Labels present", "Value": ", ".join(map(str, labels_present))},
        {"Metric": "Main notes", "Value": "; ".join(reasons[:3])},
    ]
    return pd.DataFrame(rows)


def make_summary_html(reasons, channel_table, label_table, sampling_info, df):
    best_channels = channel_table.sort_values("Activity score", ascending=False)["Channel"].head(3).tolist()
    active_labels = sorted(set(label_table["Label"].astype(int).tolist()).intersection({1, 2, 3, 4, 5, 6}))
    initial_rows = df.attrs.get("initial_rows", len(df))
    dropped_rows = df.attrs.get("dropped_rows", 0)
    notes_html = "".join([f"<li>{r}</li>" for r in reasons[:5]])
    if not notes_html:
        notes_html = "<li>No major issues detected by the current checks.</li>"

    html = f"""
    <div style="border:1px solid #d1d5db; background:#ffffff; border-left:7px solid #b91c1c; padding:16px 18px; border-radius:12px;">
        <div style="font-size:22px; font-weight:700; color:#111111; margin-bottom:8px;">Signal scan summary</div>
        <div><b>Most active channels:</b> {', '.join(best_channels) if best_channels else 'unknown'}</div>
        <div><b>Main gesture labels present:</b> {len(active_labels)}/6 ({', '.join(map(str, active_labels)) if active_labels else 'none'})</div>
        <div><b>Time gaps:</b> {sampling_info['gap_count']} gaps, {sampling_info['gap_percent']:.2f}% of intervals</div>
        <div><b>Rows removed during cleaning:</b> {dropped_rows:,} / {initial_rows:,}</div>
        <div style="margin-top:10px;"><b>Detected findings:</b></div>
        <ul style="margin-top:6px;">{notes_html}</ul>
    </div>
    """
    return html

def plot_signal_preview(df, max_points=5000):
    n = min(len(df), max_points)
    t = df["time"].iloc[:n].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    # Normalize each channel for clean stacked display.
    offset = 0.0
    yticks = []
    yticklabels = []
    for i, ch in enumerate(CHANNELS):
        x = df[ch].iloc[:n].to_numpy(dtype=float)
        scale = np.percentile(np.abs(x), 95) + 1e-12
        y = x / scale + offset
        ax.plot(t, y, linewidth=0.8, color="#111111", alpha=0.75 if i % 2 else 0.95)
        yticks.append(offset)
        yticklabels.append(ch)
        offset += 3.0
    ax.set_title("EMG signal preview, first samples", fontsize=16, fontweight="bold", color="#111111")
    ax.set_xlabel("Time")
    ax.set_ylabel("Channels, normalized and stacked")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#111111")
    ax.spines["bottom"].set_color("#111111")
    plt.tight_layout()
    return fig


def plot_channel_activity(channel_table):
    plot_df = channel_table.sort_values("Activity score", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.barh(plot_df["Channel"], plot_df["Activity score"], color="#b91c1c", edgecolor="#111111", linewidth=0.7)
    for bar, value in zip(bars, plot_df["Activity score"]):
        ax.text(value + 1, bar.get_y() + bar.get_height()/2, f"{value:.1f}", va="center", fontsize=12)
    ax.set_title("Channel activity score", fontsize=16, fontweight="bold", color="#111111")
    ax.set_xlabel("Relative activity, % of max channel")
    ax.set_xlim(0, max(105, float(plot_df["Activity score"].max()) + 10))
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_label_distribution(label_table):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    labels = [str(x) for x in label_table["Label"]]
    values = label_table["Samples"].to_numpy(dtype=float)
    ax.bar(labels, values, color="#111111", edgecolor="#b91c1c", linewidth=1.0)
    ax.set_title("Label distribution", fontsize=16, fontweight="bold", color="#111111")
    ax.set_xlabel("Gesture label")
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_diagnostic_components(sampling_info, channel_table, label_table, df):
    """Visual diagnostic indicators without assigning an overall quality score."""
    gap_clean = max(0.0, 100.0 - min(100.0, sampling_info["gap_percent"] * 20.0))
    spike_mean = float(channel_table["spike_percent"].mean()) if len(channel_table) else 0.0
    spike_clean = max(0.0, 100.0 - min(100.0, spike_mean * 30.0))
    flatline_channels = int((channel_table["flatline_percent"] > 95).sum()) if len(channel_table) else 0
    channel_ok = max(0.0, 100.0 - flatline_channels * 20.0)
    active_labels = len(set(label_table["Label"].astype(int).tolist()).intersection({1, 2, 3, 4, 5, 6})) if len(label_table) else 0
    label_coverage = 100.0 * active_labels / 6.0
    initial_rows = df.attrs.get("initial_rows", len(df))
    dropped_rows = df.attrs.get("dropped_rows", 0)
    clean_rows = 100.0 * (initial_rows - dropped_rows) / max(initial_rows, 1)

    components = pd.DataFrame({
        "Indicator": ["Clean rows", "Regular timing", "Low spike rate", "Channel continuity", "Label coverage"],
        "Percent": [clean_rows, gap_clean, spike_clean, channel_ok, label_coverage],
    })

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.bar(components["Indicator"], components["Percent"], color="#b91c1c", edgecolor="#111111", linewidth=0.8)
    for bar, val in zip(bars, components["Percent"]):
        ax.text(bar.get_x()+bar.get_width()/2, val+2, f"{val:.0f}%", ha="center", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percent")
    ax.set_title("Diagnostic indicators", fontsize=16, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=10)
    plt.tight_layout()
    return fig

def analyze_emg(file_obj):
    try:
        file_path = get_file_path(file_obj)
        df = read_emg_file(file_path)
        sampling_info = estimate_sampling(df)
        channel_table = compute_channel_metrics(df)
        X, y_win, times, purities = compute_window_features(df)
        label_table = compute_label_metrics(df, X, y_win, purities)
        suspicious_table = detect_suspicious_intervals(df, sampling_info)
        _quality_score, reasons = compute_quality_score(df, sampling_info, channel_table, label_table)
        overview_table = build_overview(df, sampling_info, reasons, file_path)
        summary_html = make_summary_html(reasons, channel_table, label_table, sampling_info, df)
        signal_plot = plot_signal_preview(df)
        channel_plot = plot_channel_activity(channel_table)
        label_plot = plot_label_distribution(label_table)
        quality_plot = plot_diagnostic_components(sampling_info, channel_table, label_table, df)
        status = f"Analysis finished successfully. File: {Path(file_path).name}. Valid samples: {len(df):,}."
        return (
            status,
            summary_html,
            overview_table,
            channel_table,
            label_table,
            suspicious_table,
            quality_plot,
            signal_plot,
            channel_plot,
            label_plot,
        )
    except Exception as e:
        tb = traceback.format_exc()
        empty = pd.DataFrame()
        return (
            f"Analysis error: {e}",
            f"<div style='color:#991b1b; border:1px solid #fecaca; padding:12px; border-radius:8px;'><b>Error:</b> {e}<br><pre>{tb}</pre></div>",
            empty,
            empty,
            empty,
            empty,
            None,
            None,
            None,
            None,
        )


with gr.Blocks(title="EMG Signal Quality Analyzer") as demo:
    gr.Markdown(
        """
        <div id="title-block">
            <h1>EMG Signal Quality Analyzer</h1>
            <p>Upload a raw EMG file to review timing, channel activity, label coverage and suspicious signal intervals.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload raw EMG file (.txt or .csv)", file_types=[".txt", ".csv"])
            analyze_btn = gr.Button("Analyze EMG Signal", variant="primary")
            gr.Markdown(
                """
                **Expected columns:** `time ch1 ch2 ch3 ch4 ch5 ch6 ch7 ch8 label`  
                If no file is uploaded, the built-in demo file will be analyzed.
                """
            )
        with gr.Column(scale=2):
            status_box = gr.Textbox(label="Status", lines=2)
            summary_box = gr.HTML(label="Signal summary")

    with gr.Tab("Overview"):
        overview_df = gr.Dataframe(label="File overview", wrap=True)
        quality_plot = gr.Plot(label="Diagnostic indicators")

    with gr.Tab("Signal Preview"):
        signal_plot = gr.Plot(label="Stacked EMG preview")

    with gr.Tab("Channels"):
        channel_df = gr.Dataframe(label="Channel metrics", wrap=True)
        channel_plot = gr.Plot(label="Channel activity")

    with gr.Tab("Gesture Labels"):
        label_df = gr.Dataframe(label="Label distribution and valid windows", wrap=True)
        label_plot = gr.Plot(label="Label distribution")

    with gr.Tab("Suspicious Intervals"):
        suspicious_df = gr.Dataframe(label="Detected gaps and amplitude spikes", wrap=True)

    analyze_btn.click(
        fn=analyze_emg,
        inputs=[file_input],
        outputs=[
            status_box,
            summary_box,
            overview_df,
            channel_df,
            label_df,
            suspicious_df,
            quality_plot,
            signal_plot,
            channel_plot,
            label_plot,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True, show_error=True, css=CSS, theme=gr.themes.Base())
