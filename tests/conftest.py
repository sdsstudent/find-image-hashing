"""Test setup — ensure `find_image_hashing` is importable.

Two paths to make this work:
  1. `pip install -e .` from repo root (recommended) — package is installed,
     no path manipulation needed; this conftest becomes a no-op.
  2. Without install — fall back to sys.path manipulation pointing at src/.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
# Insert src/ at the front of sys.path so `import find_image_hashing` resolves
# even without `pip install -e .`. Idempotent if the path is already present.
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# Repo root for tests that still touch loose top-level scripts (e.g. profile_script.py).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
