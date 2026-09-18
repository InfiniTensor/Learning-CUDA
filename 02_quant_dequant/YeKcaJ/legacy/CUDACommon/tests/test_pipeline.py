"""兼容旧 CTest 路径，实际测试位于 Core/tests。"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "Core/tests/test_pipeline.py"), run_name="__main__")
