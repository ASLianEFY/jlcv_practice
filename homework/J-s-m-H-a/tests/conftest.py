"""Make the homework module importable from the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

HOMEWORK_DIR = Path(__file__).resolve().parents[1]
if str(HOMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(HOMEWORK_DIR))
