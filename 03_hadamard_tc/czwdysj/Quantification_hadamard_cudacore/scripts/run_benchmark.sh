#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode=representative
[[ "${1:-}" == "--full" ]] && mode=full
stamp="$(date -u +%Y%m%dT%H%M%SZ)_${mode}"
out="$root/results/$stamp"; mkdir -p "$out"
cmake -S "$root" -B "$root/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/build" -j"$(nproc)"
ctest --test-dir "$root/build" --output-on-failure
: > "$out/correctness.log"
for dim in 128 1024 8192; do for pattern in random basis sequence zero; do
  "$root/build/benchmark_cuda" --correctness-only --batch 4 --dim "$dim" --dtype all --pattern "$pattern" >> "$out/correctness.log"
done; done
if [[ "$mode" == full ]]; then
  shapes=(); for batch in 1 32 128 1024 4096; do for dim in 128 256 512 1024 2048 4096 8192; do shapes+=("$batch:$dim"); done; done
else
  shapes=(32:128 32:256 32:512 32:1024 32:2048 32:4096 32:8192 1:1024 128:1024 1024:1024 4096:1024 4096:8192)
fi
raw="$out/raw.csv"
for shape in "${shapes[@]}"; do batch="${shape%:*}"; dim="${shape#*:}"
  "$root/build/benchmark_cuda" --version all --batch "$batch" --dim "$dim" --dtype all --warmup 100 --repeat 1000 --trials 3 --csv "$raw"
done
python3 "$root/benchmark/report_results.py" "$raw" "$out/report.md" --summary "$out/summary.csv"
python3 - "$out/metadata.json" "$mode" <<'PY'
import json,subprocess,sys,datetime
def cmd(x):
 try:return subprocess.check_output(x,text=True,stderr=subprocess.STDOUT).strip()
 except Exception as e:return str(e)
data={"timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"mode":sys.argv[2],
"gpu":cmd(["nvidia-smi","--query-gpu=name,compute_cap,driver_version","--format=csv,noheader"]),
"nvcc":cmd(["nvcc","--version"]),"ncu":cmd(["ncu","--version"])}
open(sys.argv[1],"w").write(json.dumps(data,indent=2)+"\n")
PY
echo "Results: $out"
