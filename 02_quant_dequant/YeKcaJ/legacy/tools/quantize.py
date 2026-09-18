#!/usr/bin/env python3
"""旧命令兼容入口；请阅读和修改 Core/tools/quantize.py。"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "Core/tools/quantize.py"), run_name="__main__")
