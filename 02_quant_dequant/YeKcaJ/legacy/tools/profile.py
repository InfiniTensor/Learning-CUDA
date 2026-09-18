#!/usr/bin/env python3
"""旧命令兼容入口；正式采集使用 Core/build 中的程序。"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "Core/tools/profile.py"), run_name="__main__")
