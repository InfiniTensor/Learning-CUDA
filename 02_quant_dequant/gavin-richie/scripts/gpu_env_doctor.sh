#!/usr/bin/env bash
# GPU environment doctor: diagnose the three failure modes seen in shared
# GPU containers before they surface as cryptic CUDA errors.
#
#   1. device nodes missing   /dev/nvidia[0-9]*, /dev/nvidia-uvm not injected
#                             (host released the GPU) -> cuInit fails 100/999
#   2. driver lib is a stub   libcuda.so.<ver> / libnvidia-ml.so.<ver> mounted
#                             as 0-byte placeholders -> dlopen "file too short"
#   3. version mixing         old cuda-compat userspace lib (e.g. 570) loaded
#                             against a newer kernel module (e.g. 610)
#                             -> cuInit fails 803 "unsupported driver combo"
#
# Usage:
#   bash scripts/gpu_env_doctor.sh            # diagnose, exit 0 = healthy
#   bash scripts/gpu_env_doctor.sh --fix      # additionally print eval-able
#                                             # repairs for fixable cases
#
# Exit codes: 0 healthy, 1 broken, 2 degraded (usable with warnings).
set -uo pipefail

FIX=0
[ "${1:-}" = "--fix" ] && FIX=1

fail=0; warn=0
say()  { printf '%s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; fail=1; }
warn_(){ printf '  [WARN] %s\n' "$*"; warn=1; }
info() { printf '  [ inf ] %s\n' "$*"; }
ok()   { printf '  [ ok ] %s\n' "$*"; }

STUB_MAX_BYTES=4096   # real driver libs are tens of MB

# --- 1. kernel module & device nodes --------------------------------------
say "== kernel module & device nodes =="
ksm_ver=""
if [ -r /proc/driver/nvidia/version ]; then
  ksm_ver=$(grep -m1 'NVRM version' /proc/driver/nvidia/version \
            | grep -oE '[0-9]{3,}\.[0-9]{2}\.[0-9]{2}' | head -1)
  ok "kernel module (NVRM) present, version ${ksm_ver:-unknown}"
else
  bad "/proc/driver/nvidia/version missing - no NVIDIA kernel driver"
fi

gpu_nodes=$(ls /dev/nvidia[0-9]* 2>/dev/null | head -1)
if [ -n "$gpu_nodes" ]; then
  ok "GPU device node present: $(ls /dev/nvidia[0-9]* 2>/dev/null | tr '\n' ' ')"
else
  bad "/dev/nvidia[0-9]* missing - host did not inject (or released) the GPU"
fi
if [ -e /dev/nvidia-uvm ]; then
  ok "/dev/nvidia-uvm present"
else
  bad "/dev/nvidia-uvm missing - CUDA init will fail even with libcuda intact"
fi
[ -e /dev/nvidiactl ] && ok "/dev/nvidiactl present" \
                       || bad "/dev/nvidiactl missing"

# --- 2. userspace driver libraries ----------------------------------------
say "== userspace driver libraries =="
# classify every libcuda.so.* / libnvidia-ml.so.* regular file in the usual
# locations; report size, and whether the version matches the kernel module.
classify_lib() { # $1 = file path (may be a symlink; the environment's stat
                 # does not follow links, so resolve first)
  local f=$1 base ver size
  f=$(readlink -f "$f" 2>/dev/null) || return 3
  [ -f "$f" ] || return 3
  base=$(basename "$f")
  ver=${base#libcuda.so.}; ver=${ver#libnvidia-ml.so.}
  size=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$size" -le "$STUB_MAX_BYTES" ]; then
    bad "$base is a STUB ($size bytes) at $f - dlopen gives 'file too short'"
    return 1
  fi
  if [ -n "$ksm_ver" ] && [ "$ver" != "$ksm_ver" ]; then
    # on disk but not in the load path is fine (live probe decides); it only
    # becomes a real problem via LD_LIBRARY_PATH, checked separately below.
    info "$base ($((size/1024/1024)) MB) version $ver != kernel module $ksm_ver (not in load path)"
    return 2
  fi
  ok "$base real ($((size/1024/1024)) MB) at $f, matches kernel module"
  return 0
}

best_cuda_lib=""
for f in $( { ls /usr/lib/x86_64-linux-gnu/libcuda.so.* \
                 /usr/lib64/libcuda.so.* \
                 /usr/local/cuda*/compat/lib.real/libcuda.so.* 2>/dev/null; } \
             | while read -r p; do readlink -f "$p"; done | sort -u ); do
  [ -f "$f" ] || continue
  classify_lib "$f"; rc=$?
  # remember the first real, version-matching lib as the preferred one
  [ $rc -eq 0 ] && [ -z "$best_cuda_lib" ] && best_cuda_lib=$f
done
for f in $( ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.* 2>/dev/null \
            | while read -r p; do readlink -f "$p"; done | sort -u ); do
  [ -f "$f" ] || continue
  classify_lib "$f"
done

# --- 3. LD_LIBRARY_PATH sanity --------------------------------------------
say "== LD_LIBRARY_PATH sanity =="
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  IFS=':' read -ra dirs <<< "$LD_LIBRARY_PATH"
  for d in "${dirs[@]}"; do
    [ -n "$d" ] || continue
  resolved=$(ls "$d"/libcuda.so.* 2>/dev/null \
             | while read -r p; do readlink -f "$p"; done | sort -u)
  for f in $resolved; do
    [ -f "$f" ] || continue
    size=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$size" -le "$STUB_MAX_BYTES" ]; then
      bad "LD_LIBRARY_PATH entry shadows libcuda with a stub: $f"
      if [ $FIX -eq 1 ] && [ -n "$best_cuda_lib" ]; then
        say "    fix: drop '$d' from LD_LIBRARY_PATH (real lib is mounted elsewhere)"
      fi
    else
      ver=$(basename "$f"); ver=${ver#libcuda.so.}
      if [ -n "$ksm_ver" ] && [ "$ver" != "$ksm_ver" ] && [ -n "$best_cuda_lib" ]; then
        warn_ "LD_LIBRARY_PATH entry '$d' provides libcuda $ver (kernel $ksm_ver) -> cuInit 803"
        if [ $FIX -eq 1 ]; then
          say "    fix: remove '$d' from LD_LIBRARY_PATH"
        fi
      fi
    fi
  done
  done
else
  ok "LD_LIBRARY_PATH empty (system defaults)"
fi

# --- 4. live probe: dlopen + cuInit ----------------------------------------
say "== live probe =="
python3 - "$ksm_ver" <<'PYEOF'
import ctypes, glob, os, sys

ksm = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    cuda = ctypes.CDLL("libcuda.so.1")
    loaded = [l.split()[-1] for l in open("/proc/self/maps")
              if "libcuda.so" in l]
    print(f"  [ ok ] dlopen(libcuda.so.1) -> {loaded[-1] if loaded else '?'}")
except OSError as e:
    print(f"  [FAIL] dlopen(libcuda.so.1): {e}")
    sys.exit(1)

cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int
rc = cuda.cuInit(0)
names = {0: "ok", 100: "NO_DEVICE", 101: "INVALID_DEVICE", 999: "UNEXPECTED",
         803: "SYSTEM_DRIVER_MISMATCH (userspace libcuda / kernel combo)"}
if rc == 0:
    n = ctypes.c_int(-1)
    cuda.cuDeviceGetCount(ctypes.byref(n))
    print(f"  [ ok ] cuInit -> 0, {n.value} device(s)"
          + (f" (kernel module {ksm})" if ksm else ""))
else:
    print(f"  [FAIL] cuInit -> {rc} ({names.get(rc, 'unknown')})")
    sys.exit(1)
PYEOF
probe_rc=$?

# nvidia-smi (needs libnvidia-ml, i.e. fails with stubs too)
if command -v nvidia-smi >/dev/null 2>&1; then
  if out=$(nvidia-smi -L 2>&1) && [ -n "$out" ]; then
    ok "nvidia-smi -L: $(echo "$out" | head -1)"
  else
    bad "nvidia-smi present but silent/failing (usually libnvidia-ml stub)"
  fi
else
  warn_ "nvidia-smi not in PATH"
fi

# --- verdict ---------------------------------------------------------------
say "== verdict =="
[ $probe_rc -ne 0 ] && fail=1   # dlopen/cuInit failing is never just a warning
if [ $fail -eq 1 ]; then
  say "BROKEN: see [FAIL] items above."
  say "  - nodes/libs released by the host -> re-inject (restart container or"
  say "    nvidia-container-cli) ; nothing inside the container can fix that."
  [ $FIX -eq 1 ] && say "  - re-run with the printed 'fix:' lines applied to your environment."
  exit 1
elif [ $warn -eq 1 ] || [ $probe_rc -ne 0 ]; then
  say "DEGRADED: usable but with the [WARN] items above."
  exit 2
else
  say "HEALTHY: kernel module, device nodes, driver libs and cuInit all consistent."
  exit 0
fi
