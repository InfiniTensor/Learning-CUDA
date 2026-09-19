#!/usr/bin/env bash
set -euo pipefail

cli="$1"
rows="${2:-1024}"
cols="${3:-2048}"
workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT

input_bytes=$((rows * cols * 4))
dd if=/dev/zero of="${workdir}/input.bin" bs=4 count="$((rows * cols))" status=none
start_ns="$(date +%s%N)"
"${cli}" quantize --input "${workdir}/input.bin" --output "${workdir}/tensor.qnt" \
  --rows "${rows}" --cols "${cols}" --input-type fp32 --format nvfp4 >/dev/null
quant_end_ns="$(date +%s%N)"
"${cli}" dequantize --input "${workdir}/tensor.qnt" --output "${workdir}/output.bin" \
  --output-type bf16 >/dev/null
end_ns="$(date +%s%N)"
quant_ms=$(( (quant_end_ns - start_ns) / 1000000 ))
dequant_ms=$(( (end_ns - quant_end_ns) / 1000000 ))
total_ms=$(( (end_ns - start_ns) / 1000000 ))
payload_bytes="$(stat -c '%s' "${workdir}/tensor.qnt")"
output_bytes="$(stat -c '%s' "${workdir}/output.bin")"
printf 'rows,cols,input_bytes,quantize_ms,dequantize_ms,total_ms,quantized_file_bytes,output_bytes\n'
printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
  "${rows}" "${cols}" "${input_bytes}" "${quant_ms}" "${dequant_ms}" \
  "${total_ms}" "${payload_bytes}" "${output_bytes}"
