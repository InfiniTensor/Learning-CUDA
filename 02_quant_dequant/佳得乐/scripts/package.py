#!/usr/bin/env python3
"""打包源码与轻量实测证据；不打包构建目录或可再生的大矩阵。"""
import hashlib,json,pathlib,tarfile
root=pathlib.Path(__file__).resolve().parents[1]
run=root/'results/full_release'
if not (run/'SUCCESS').is_file():raise RuntimeError('full_release has not passed')
meta=json.loads((run/'metadata.json').read_text())
for name,digest in meta['source_sha256'].items():
 if name.startswith(('src/','include/','tests/','configs/')):
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('tested source changed: '+name)
files=[]
for folder in ['include','src','tests','configs','scripts','docs']:
 files.extend(p for p in (root/folder).glob('*') if p.is_file())
files.extend(root/n for n in ['README.md','CMakeLists.txt','.gitignore','.clang-format'])
for name in ['full_release','quick_release','baseline','environment']:
 for p in (root/'results'/name).rglob('*'):
  if p.is_file() and p.suffix in ['.log','.txt','.md','.csv','.json','.svg','.png'] and 'test-artifacts' not in p.parts and 'data' not in p.relative_to(root/'results'/name).parts:
   files.append(p)
 if (root/'results'/name/'SUCCESS').is_file():files.append(root/'results'/name/'SUCCESS')
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
(root/'results/delivery-manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(root/'results/delivery.tar.gz','w:gz') as tar:
 for p in sorted(set(files)):tar.add(p,arcname='software/'+str(p.relative_to(root)))
 tar.add(root/'results/delivery-manifest.json',arcname='software/delivery-manifest.json')
with tarfile.open(root/'results/delivery.tar.gz','r:gz') as tar:
 for name,digest in manifest.items():
  f=tar.extractfile('software/'+name)
  if hashlib.sha256(f.read()).hexdigest()!=digest:raise RuntimeError('archive verification failed: '+name)
print('PASS tested core hashes and archive manifest;',len(manifest),'files;', (root/'results/delivery.tar.gz').stat().st_size,'bytes')
