#!/usr/bin/env bash
# One-command venv setup for ncaarotamers.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo
echo "ncaarotamers env ready."
echo "Activate with:  source venv/bin/activate"
echo "Then run e.g.:  python ncaarotamers.py --cif cifs/VAL.cif --tricode VAL --denovo"
