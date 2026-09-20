"""按 Learning-CUDA 的提交流程, 把本目录内容整理成一份可直接放进
`<选题>/<ID>/` 的干净副本。

排除规则与仓库根的 `.gitignore` 对齐 (构建产物、可再生的数据文件、剖析报告), 避免把
`build/` (约 59 MB) 与 `data/tensors/`、`data/out/*.bin` (合计约 4.9 GB) 一起打进去。

    python scripts/pack_submission.py --id <报名ID> --dest <fork目录>/02_quant_dequant/<报名ID>

`--dest` 不存在时会创建; 已存在同名文件会被覆盖。加 `--dry-run` 只列出会拷贝的文件与
体积, 不写盘。
"""

import argparse
import fnmatch
import io
import os
import shutil
import sys

# 与 .gitignore 对齐: (目录名, 是否整棵子树排除) / 文件名通配
SKIP_DIRS = ("build", "__pycache__", ".git", ".vscode", ".idea")
SKIP_DIRS_ALWAYS = ("data/tensors",)
SKIP_FILE_GLOBS = (
    "*.nsys-rep",
    "*.sqlite",
    "*.pyc",
    "data/out/*_lowp.bin",
    "data/out/*_dequant.bin",
)
def rel_posix(path, root):
    return os.path.relpath(path, root).replace(os.sep, "/")


def is_skipped(rel):
    parts = rel.split("/")
    if any(p in SKIP_DIRS for p in parts):
        return True
    for d in SKIP_DIRS_ALWAYS:
        if rel == d or rel.startswith(d + "/"):
            return True
    return any(fnmatch.fnmatch(rel, g) for g in SKIP_FILE_GLOBS)


def inside(child, parent):
    """child 是否在 parent 之内 (跨盘符时 commonpath 会抛错, 视为不在)。"""
    child = os.path.normcase(os.path.abspath(child))
    parent = os.path.normcase(os.path.abspath(parent))
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:
        return False


def collect(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = rel_posix(full, root)
            if is_skipped(rel):
                continue
            files.append((full, rel))
    return files


def main():
    ap = argparse.ArgumentParser(description="整理提交副本")
    ap.add_argument("--id", required=True, help="报名 ID, 仅用于打印与校验")
    ap.add_argument("--dest", required=True, help="拷贝目标目录, 例如 02_quant_dequant/<ID>")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    files = collect(root)
    total = sum(os.path.getsize(f) for f, _ in files if os.path.exists(f))

    if args.dry_run:
        for _, rel in files:
            print(rel)
        print("---")
    else:
        dest = os.path.abspath(args.dest)
        if inside(dest, root):
            sys.stderr.write("目标目录不能在本项目目录内部: %s\n" % dest)
            return 2
        for full, rel in files:
            out = os.path.join(dest, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy2(full, out)

    print("报名 ID     : %s" % args.id)
    print("文件数      : %d" % len(files))
    print("原始体积    : %.2f MB" % (total / 1048576.0))
    if not args.dry_run:
        print("已写入      : %s" % os.path.abspath(args.dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
