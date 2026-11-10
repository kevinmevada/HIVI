"""Import shim so the chat app shares hivi.paths whether or not `pip install -e .` ran."""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from hivi import paths
except ImportError:  # running straight from a checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from hivi import paths

__all__ = ["paths"]
