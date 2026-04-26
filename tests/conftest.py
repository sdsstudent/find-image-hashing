"""Ensure the repo root is importable from tests so `from FINd import ...` works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
