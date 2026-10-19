"""C2 — CLI 冒烟测试。

验证 main.py 可执行、--help 正常退出，并列出 --video / --out-dir /
--no-frames / --trail / --debug 参数。
"""

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
