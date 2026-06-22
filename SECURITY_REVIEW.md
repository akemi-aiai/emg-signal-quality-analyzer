# Security Review

This app was designed as a local deterministic EMG data analyzer.

## What the app does

- Reads `.txt` or `.csv` EMG files.
- Converts columns to numeric values.
- Computes deterministic statistics and plots.
- Runs locally on `127.0.0.1:7860`.

## What the app does not do

- No `eval`.
- No `exec`.
- No `os.system`.
- No `subprocess`.
- No `pickle.load`.
- No `joblib.load`.
- No saved ML model loading.
- No `share=True` public Gradio link.

## File safety limits

- Allowed file extensions: `.txt`, `.csv`.
- Maximum file size: 50 MB.
- Maximum rows read: 1,000,000.

## Remaining limitations

Uploaded files are processed as numeric tables. The application is intended for trusted local use, not as a public web service.
