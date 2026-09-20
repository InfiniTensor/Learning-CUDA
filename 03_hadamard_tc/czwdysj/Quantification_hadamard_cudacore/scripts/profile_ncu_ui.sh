#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ncu_bin="${NCU_BIN:-/usr/local/cuda/bin/ncu}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="${1:-$root/results/ncu_ui_final_$stamp}"
mkdir -p "$out"
validation="$out/validation.log"
: > "$validation"

cmake -S "$root" -B "$root/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/build" -j"$(nproc)"

echo "Profile configuration: FP16 [1024,4096], warmup=100, measured_launches=1" | tee -a "$validation"
for version in v0 v1 v2; do
  name="fused_${version}_fp16_b1024_d4096"
  echo "Collecting $name" | tee -a "$validation"
  "$ncu_bin" --profile-from-start off --target-processes all \
    --set full --launch-count 1 --import-source yes \
    --source-folders "$root" --force-overwrite --export "$out/$name" \
    "$root/build/benchmark_cuda" --version "$version" --batch 1024 --dim 4096 \
    --dtype fp16 --warmup 100 --repeat 1 --trials 1 --profile \
    >> "$validation" 2>&1

  report="$out/$name.ncu-rep"
  test -s "$report"
  echo "Linux import validation: $report" | tee -a "$validation"
  "$ncu_bin" --import "$report" --page details --print-details all \
    > "$out/${name}_linux_details.txt" 2>> "$validation"
  grep -q "fused_${version}_kernel" "$out/${name}_linux_details.txt"
  grep -q "Block Size.*256" "$out/${name}_linux_details.txt"
  grep -Eq "Grid Size[[:space:]]+1024" "$out/${name}_linux_details.txt"
done

python3 - "$out" "$ncu_bin" <<'PY'
import hashlib,json,pathlib,subprocess,sys,datetime
out=pathlib.Path(sys.argv[1]); ncu=sys.argv[2]
def command(args):
    try:return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()
    except Exception as e:return f"ERROR: {e}"
reports=[]
for path in sorted(out.glob("*.ncu-rep")):
    reports.append({"file":path.name,"bytes":path.stat().st_size,
                    "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
manifest={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "workload":{"dtype":"fp16","batch":1024,"dim":4096,"warmup":100,"measured_launches":1},
 "gpu":command(["nvidia-smi","--query-gpu=name,compute_cap,driver_version","--format=csv,noheader"]),
 "cuda":command(["nvcc","--version"]),"linux_ncu":command([ncu,"--version"]),"reports":reports}
(out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
PY

cat > "$out/README.md" <<EOF
# Nsight Compute UI reports

Workload: FP16 input, batch 1024, dimension 4096. Each report excludes allocation,
initialization, correctness work, and 100 synchronized warmup launches. It contains
one measured launch; profiler replay passes are part of NCU collection, not extra
benchmark samples.

Open from Windows Explorer by double-clicking a report, or run:

\`ncu-ui.bat "<Windows path to fused_v2_fp16_b1024_d4096.ncu-rep>"\`

Reports were collected with \`--set full --import-source yes\`; CUDA source,
line information, PTX/SASS, full supported sections, and rule diagnostics are stored
inside each report. Checksums and environment versions are in \`manifest.json\`.

Regenerate from WSL:

\`./scripts/profile_ncu_ui.sh\`
EOF

echo "Linux collection and import validation passed: $out" | tee -a "$validation"
if command -v powershell.exe >/dev/null 2>&1; then
  script_win="$(wslpath -w "$root/scripts/validate_ncu_ui_windows.ps1")"
  out_win="$(wslpath -w "$out")"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$script_win" -ReportDirectory "$out_win"
  echo "Windows CLI and NCU-UI validation passed" | tee -a "$validation"
else
  echo "Windows validation skipped: powershell.exe unavailable" | tee -a "$validation"
  exit 3
fi
echo "$out"
