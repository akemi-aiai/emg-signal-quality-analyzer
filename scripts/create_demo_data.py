from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data_examples" / "demo_emg_signal.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
fs = 1000
channels = [f"ch{i}" for i in range(1, 9)]

segments = []
current_t = 0
# label, seconds, main active channels, amplitude
config = [
    (1, 1.8, [0, 1], 0.00004),
    (2, 2.0, [3, 4, 6], 0.00022),
    (3, 1.7, [2, 5, 7], 0.00018),
    (4, 1.8, [0, 3, 4], 0.00020),
    (5, 1.6, [4, 6], 0.00017),
    (6, 1.7, [1, 5, 7], 0.00019),
]

rows = []
for label, seconds, active_ch, amp in config:
    n = int(seconds * fs)
    time = np.arange(current_t, current_t + n)
    base = rng.normal(0, 0.000025, size=(n, 8))
    for ch in active_ch:
        carrier = np.sin(2 * np.pi * rng.uniform(40, 120) * np.arange(n) / fs)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n) / (n + 1))
        base[:, ch] += amp * envelope * carrier + rng.normal(0, amp * 0.25, size=n)
    # Add a couple deterministic spikes in active gesture region
    if label in {2, 4}:
        idx = n // 2
        base[idx, active_ch[0]] += amp * 12
    for i in range(n):
        rows.append([time[i], *base[i].tolist(), label])
    current_t += n
    # Add short time gap between gestures
    current_t += 15

df = pd.DataFrame(rows, columns=["time"] + channels + ["label"])
df.to_csv(OUT, sep=" ", index=False)
print(f"Saved {OUT} with shape {df.shape}")
