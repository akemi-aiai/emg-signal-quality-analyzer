from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
py_files = [p for p in ROOT.rglob("*.py") if ".venv" not in p.parts]
forbidden = [
    "ev" + "al(",
    "ex" + "ec(",
    "os." + "system",
    "sub" + "process",
    "pickle." + "load",
    "joblib." + "load",
    "share" + "=True",
]

bad = []
for p in py_files:
    if p.name == "check_safety.py":
        continue
    text = p.read_text(encoding="utf-8")
    for token in forbidden:
        if token in text:
            bad.append((p.relative_to(ROOT), token))

if bad:
    print("Potentially unsafe tokens found:")
    for p, token in bad:
        print(f"- {p}: {token}")
    raise SystemExit(1)

print("Safety check passed: no forbidden tokens found in project Python files.")
