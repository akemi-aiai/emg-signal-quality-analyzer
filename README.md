# EMG Signal Quality Analyzer

A local Gradio application for scanning raw EMG recordings and extracting useful information about the signal.

The app reads an EMG file and shows:

- sampling-rate estimate and time gaps;
- missing or invalid rows after numeric cleaning;
- channel activity, RMS, MAV and amplitude range;
- possible flatline channels;
- amplitude spikes by channel;
- gesture label coverage and valid 200 ms windows;
- diagnostic indicators for rows, timing, spikes, channel continuity and labels.

## Expected input format

The uploaded file should be `.txt` or `.csv` with 10 columns:

```text
time ch1 ch2 ch3 ch4 ch5 ch6 ch7 ch8 label
```

The reader supports files with or without a header row, and whitespace, comma or semicolon separators.

If no file is uploaded, the app analyzes the built-in demo file:

```text
data_examples/demo_emg_signal.txt
```

## Run on macOS / local computer

```bash
cd emg_signal_quality_analyzer_clean_readable
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python emg_signal_quality_analyzer.py
```

Open:

```text
http://127.0.0.1:7860
```

## Product idea

This application helps check whether an EMG recording is suitable for calibration and gesture-analysis work. It focuses on measurable signal properties instead of a virtual prosthesis animation.
