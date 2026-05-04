"""
run.py  —  Clean launcher for the IDS app.
Clears all Python caches first, then starts Streamlit.

Usage:  python run.py
"""

import os
import sys
import shutil
import subprocess

# ── Step 1: Locate the project folder (same folder as this script) ───────────
BASE = os.path.dirname(os.path.abspath(__file__))
print(f"[run.py] Project folder: {BASE}")

# ── Step 2: Delete ALL __pycache__ folders and .pyc files ───────────────────
deleted = []
for root, dirs, files in os.walk(BASE):
    for d in dirs:
        if d == "__pycache__":
            full = os.path.join(root, d)
            shutil.rmtree(full, ignore_errors=True)
            deleted.append(full)
    for f in files:
        if f.endswith(".pyc"):
            full = os.path.join(root, f)
            os.remove(full)
            deleted.append(full)

if deleted:
    print(f"[run.py] Cleared {len(deleted)} cache item(s):")
    for d in deleted:
        print(f"          {d}")
else:
    print("[run.py] No cache found — clean start.")

# ── Step 3: Verify predictor.py has the fix ──────────────────────────────────
predictor_path = os.path.join(BASE, "predictor.py")
if not os.path.exists(predictor_path):
    print("[run.py] ERROR: predictor.py not found in", BASE)
    sys.exit(1)

with open(predictor_path, "r", encoding="utf-8") as f:
    content = f.read()

if "_encode_for_dt" not in content:
    print("\n[run.py] !! WARNING: Your predictor.py does NOT have the fix !!")
    print("[run.py]    You are still using the OLD predictor.py.")
    print("[run.py]    Please replace it with the fixed version from Claude.")
    print("[run.py]    The fixed file has a function called '_encode_for_dt'.")
    print()
    ans = input("[run.py] Continue anyway? (y/n): ").strip().lower()
    if ans != "y":
        sys.exit(1)
else:
    print("[run.py] predictor.py verified — fix is present.")

# ── Step 4: Launch Streamlit ─────────────────────────────────────────────────
app_path = os.path.join(BASE, "app.py")
if not os.path.exists(app_path):
    print("[run.py] ERROR: app.py not found in", BASE)
    sys.exit(1)

print(f"\n[run.py] Starting Streamlit...\n{'='*50}")
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", app_path,
     "--server.port", "8501",
     "--server.runOnSave", "false"],   # disable hot-reload to prevent stale imports
    cwd=BASE
)
