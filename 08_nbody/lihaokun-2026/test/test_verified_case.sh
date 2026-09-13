#!/usr/bin/env bash
# Run one requested size at a time: bash test_verified_case.sh 4096 ./nbody_a800_v2
set -euo pipefail
count="${1:?Usage: bash test_verified_case.sh 4096|65536 ./nbody_a800_v2}"
executable="${2:-./nbody_a800_v2}"
case "$count" in 4096|65536) ;; *) echo 'Particle count must be 4096 or 65536' >&2; exit 2;; esac
out="results/verified_${count}"
stem="benchmark_${count}"
mkdir -p "$out"
"$executable" "data/${stem}_particles.txt" "data/${stem}_params.txt" \
  "$out/${stem}.bin" "$out/${stem}.performance.log"
python3 verify_trajectory.py "$out/${stem}.bin" \
  --particles "data/${stem}_particles.txt" --params "data/${stem}_params.txt" \
  --sample-particles 16 --report "$out/${stem}.validation.json"
# Export only after the physical checks above succeed. The report itself
# distinguishes PASS_SAMPLED from full independent verification.
python3 visualize.py "$out/${stem}.bin" --backend gpu --dimension 3d --layout detail \
  --objects "data/${stem}_objects.txt" --fps 25 --trail 160 --trail-particles 32 \
  --record-dt .01 --time-unit Myr --video-encoder cpu --z-scale 1 \
  --output "$out/${stem}_detail.mp4"
printf 'Viewer: trajectory_viewer.html\nBIN: %s\nReport: %s\n' \
  "$out/${stem}.bin" "$out/${stem}.validation.json"
