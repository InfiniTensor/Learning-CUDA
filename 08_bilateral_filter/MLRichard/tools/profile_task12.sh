#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/conda-release"
APP=""
INPUT="${ROOT_DIR}/artifacts/profiling/inputs/test_4k.raw"
OUTPUT_ROOT="${ROOT_DIR}/artifacts/profiling"
PARAMS="${ROOT_DIR}/params.txt"
RUN_NCU=1
RUN_NSYS=1
GENERATE_INPUT=1
NCU_SET="full"
NCU_PERMISSION_ERROR=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --build-dir <dir>     Build directory containing bilateral_filter
  --app <path>          Explicit bilateral_filter executable path
  --input <path>        Input raw file path (default: auto-generated 4K raw)
  --output-dir <dir>    Directory for profiling artifacts
  --params <path>       Parameter file path (default: params.txt)
  --skip-generate       Do not auto-generate missing input raw
  --skip-ncu            Skip Nsight Compute captures
  --skip-nsys           Skip Nsight Systems capture
  --ncu-set <name>      Nsight Compute set (default: full)
  -h, --help            Show this help
USAGE
}

is_wsl() {
  [[ -n "${WSL_DISTRO_NAME:-}" ]] || grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null
}

write_ncu_permission_hint() {
  local hint_path="${RUN_DIR}/ncu_permission_hint.txt"
  cat > "$hint_path" <<HINT
Nsight Compute could not access GPU Performance Counters.
Detected GPU: ${GPU_NAME}
Detected environment: $(if is_wsl; then echo "WSL"; else echo "Linux"; fi)

Official NVIDIA reference:
  https://developer.nvidia.com/ERR_NVGPUCTRPERM

Recommended fix for WSL + RTX 4070:
  1. On the Windows host, open NVIDIA Control Panel as Administrator.
  2. Enable: Desktop -> Enable Developer Settings.
  3. Open: Developer -> Manage GPU Performance Counters.
  4. Select: Allow access to the GPU performance counter to all users.
  5. Reopen the WSL terminal and rerun:
       source ~/miniconda3/etc/profile.d/conda.sh
       conda activate bilateral-cuda124
       bash tools/profile_task12.sh

Recommended fix for native Linux / A100 (permanent, Ubuntu/Debian):
  sudo tee /etc/modprobe.d/nvidia-prof.conf >/dev/null <<'EOCONF'
  options nvidia NVreg_RestrictProfilingToAdminUsers=0
  EOCONF
  sudo update-initramfs -u -k all
  sudo reboot

Verification after reboot:
  cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly

Temporary Linux workaround from NVIDIA (may require stopping the display manager):
  sudo systemctl isolate multi-user
  sudo modprobe -rf nvidia_uvm nvidia_drm nvidia_modeset nvidia-vgpu-vfio nvidia
  sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
  sudo systemctl isolate graphical
HINT
  printf 'Nsight Compute permission hint written to: %s\n' "$hint_path" >&2
}

run_ncu_capture() {
  local kernel_name="$1"
  local report_base="${RUN_DIR}/profile_${kernel_name}"
  local log_file="${RUN_DIR}/profile_${kernel_name}.ncu.log"

  set +e
  "$NCU_BIN" --set "$NCU_SET" --target-processes all -f -o "$report_base" \
    "$APP" "$INPUT" /dev/null "$PARAMS" "$kernel_name" 2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ $status -eq 0 ]]; then
    return 0
  fi

  if grep -q 'ERR_NVGPUCTRPERM' "$log_file"; then
    NCU_PERMISSION_ERROR=1
    printf 'Nsight Compute permission denied for kernel=%s, continuing with remaining steps.\n' "$kernel_name" >&2
    return 0
  fi

  printf 'Nsight Compute failed for kernel=%s, see %s\n' "$kernel_name" "$log_file" >&2
  return "$status"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --app) APP="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --output-dir) OUTPUT_ROOT="$2"; shift 2 ;;
    --params) PARAMS="$2"; shift 2 ;;
    --skip-generate) GENERATE_INPUT=0; shift ;;
    --skip-ncu) RUN_NCU=0; shift ;;
    --skip-nsys) RUN_NSYS=0; shift ;;
    --ncu-set) NCU_SET="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please activate the Conda environment first." >&2
  exit 1
fi

if [[ -z "$APP" ]]; then
  if [[ -x "${BUILD_DIR}/bilateral_filter" ]]; then
    APP="${BUILD_DIR}/bilateral_filter"
  elif [[ -x "${ROOT_DIR}/build-activated/bilateral_filter" ]]; then
    APP="${ROOT_DIR}/build-activated/bilateral_filter"
  else
    echo "Cannot find bilateral_filter executable." >&2
    exit 1
  fi
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d '\r')"
GPU_SLUG="$(printf '%s' "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | tr ' /' '__' | tr -cd '[:alnum:]_-')"
RUN_DIR="${OUTPUT_ROOT}/${GPU_SLUG}"
mkdir -p "$RUN_DIR" "$(dirname "$INPUT")"

NCU_BIN="${CONDA_PREFIX}/nsight-compute/2024.1.1/ncu"
NSYS_BIN="${CONDA_PREFIX}/nsight-compute/2024.1.1/host/target-linux-x64/nsys"

if [[ ! -x "$NCU_BIN" ]]; then
  echo "Nsight Compute not found: $NCU_BIN" >&2
  exit 1
fi
if [[ $RUN_NSYS -eq 1 && ! -x "$NSYS_BIN" ]]; then
  echo "Nsight Systems not found: $NSYS_BIN" >&2
  exit 1
fi
if [[ ! -f "$PARAMS" ]]; then
  echo "Parameter file not found: $PARAMS" >&2
  exit 1
fi
if [[ ! -f "$INPUT" ]]; then
  if [[ $GENERATE_INPUT -eq 0 ]]; then
    echo "Input raw does not exist: $INPUT" >&2
    exit 1
  fi
  python "${ROOT_DIR}/tools/generate_test_raw.py" "$INPUT" --width 3840 --height 2160 --channels 3
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
export CPATH="${CONDA_PREFIX}/include:${CPATH:-}"

BENCH_OUT="${RUN_DIR}/profile_output.raw"
printf 'GPU         : %s\n' "$GPU_NAME"
printf 'App         : %s\n' "$APP"
printf 'Input       : %s\n' "$INPUT"
printf 'Output dir  : %s\n' "$RUN_DIR"
printf 'Params      : %s\n' "$PARAMS"

if [[ $RUN_NCU -eq 1 ]]; then
  run_ncu_capture naive
  run_ncu_capture shared
  run_ncu_capture adaptive
  if [[ $NCU_PERMISSION_ERROR -eq 1 ]]; then
    write_ncu_permission_hint
  fi
fi

if [[ $RUN_NSYS -eq 1 ]]; then
  "$NSYS_BIN" profile --force-overwrite true -o "${RUN_DIR}/timeline" \
    "$APP" "$INPUT" "$BENCH_OUT" "$PARAMS" shared
  if [[ -f "${RUN_DIR}/timeline.nsys-rep" ]]; then
    "$NSYS_BIN" export --force-overwrite true --type sqlite \
      --output "${RUN_DIR}/timeline.sqlite" "${RUN_DIR}/timeline.nsys-rep" >/dev/null
  fi
fi

cat <<SUMMARY

Artifacts generated under:
  ${RUN_DIR}

Present files:
$(find "$RUN_DIR" -maxdepth 1 -type f -printf '  %f\n' | sort)
SUMMARY

if [[ $NCU_PERMISSION_ERROR -eq 1 ]]; then
  cat <<SUMMARY

Nsight Compute reports are not complete yet because GPU Performance Counters are restricted.
See:
  ${RUN_DIR}/ncu_permission_hint.txt

After enabling access, rerun:
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate bilateral-cuda124
  bash tools/profile_task12.sh
SUMMARY
else
  cat <<SUMMARY

You can open the reports locally with:
  ${CONDA_PREFIX}/nsight-compute/2024.1.1/ncu-ui ${RUN_DIR}/profile_shared.ncu-rep
SUMMARY
fi
