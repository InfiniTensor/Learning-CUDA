#!/usr/bin/env python3
"""只清理明确命名的项目产物；必须给出实际 run_id 或 --build。"""
import argparse,pathlib,shutil
root=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run-id');p.add_argument('--build',action='store_true');a=p.parse_args()
paths=[]
if a.run_id:
 if pathlib.Path(a.run_id).name!=a.run_id or a.run_id in ('.','..'):p.error('invalid run_id')
 path=root/'results'/a.run_id
 if not (path/'metadata.json').is_file():p.error('not a managed run directory')
 paths.append(path)
if a.build:paths.extend([root/'build',root/'build-cpu'])
for path in paths:
 if path.is_symlink():p.error('refuse symlink')
 if path.exists():shutil.rmtree(path)
