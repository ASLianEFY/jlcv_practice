"""C2 smoke test: the CLI entry point is importable and --help works."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "main.py"


def test_help_exits_zero_and_lists_options() -> None:
    result = subprocess.run(
        [sys.executable, str(MAIN), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for option in ("--video", "--out-dir", "--no-frames", "--trail", "--debug"):
        assert option in result.stdout
