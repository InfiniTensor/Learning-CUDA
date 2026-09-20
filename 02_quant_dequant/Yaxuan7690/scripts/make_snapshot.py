"""把整个项目目录打包成一份归档快照 (zip), 并在旁边写一个同名 `.md5.txt` 校验文件。

排除规则与 `.gitignore` / `scripts/pack_submission.py` 完全一致 (构建产物、可再生的
数据文件、剖析报告), 所以快照里既不会出现 `build/` (约 59 MB)、也不会出现
`data/tensors/` 与 `data/out/*.bin` (合计约 4.9 GB)。

    python scripts/make_snapshot.py                      # 默认写到同级目录, 名字带日期
    python scripts/make_snapshot.py --dest out.zip
    python scripts/make_snapshot.py --dry-run            # 只列清单与体积

快照存的是"当时这一刻"的目录内容, 所以**每次改完文档都要重新生成一次**, 让归档与仓库
当前内容对齐; `.md5.txt` 里的校验值是给外部备份/传输后比对用的。
"""

import argparse
import datetime
import fnmatch
import hashlib
import os
import sys
import zipfile

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
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    stamp = datetime.date.today().strftime("%Y%m%d")
    default_dest = os.path.join(os.path.dirname(root), "%s_快照_%s.zip" % (os.path.basename(root), stamp))

    ap = argparse.ArgumentParser(description="生成归档快照")
    ap.add_argument("--root", default=root)
    ap.add_argument("--dest", default=default_dest)
    ap.add_argument("--dry-run", action="store_true", help="只列清单, 不写盘")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    files = collect(root)
    total = sum(os.path.getsize(f) for f, _ in files)

    if args.dry_run:
        for _, rel in files:
            print(rel)
        print("---")
        print("文件数      : %d" % len(files))
        print("原始体积    : %.2f MB" % (total / 1048576.0))
        return 0

    dest = os.path.abspath(args.dest)
    if os.path.exists(dest):
        os.remove(dest)
    prefix = os.path.basename(root) + "/"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for full, rel in files:
            z.write(full, arcname=prefix + rel)

    with open(dest, "rb") as f:
        blob = f.read()
    digest = hashlib.md5(blob).hexdigest()

    sidecar = dest + ".md5.txt"
    with open(sidecar, "w", encoding="utf-8", newline="\n") as f:
        f.write("%s  %s\n" % (digest, os.path.basename(dest)))

    print("快照        : %s" % dest)
    print("文件数      : %d" % len(files))
    print("原始体积    : %.2f MB" % (total / 1048576.0))
    print("压缩后      : %.2f MB" % (len(blob) / 1048576.0))
    print("md5         : %s" % digest)
    print("校验文件    : %s" % sidecar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
