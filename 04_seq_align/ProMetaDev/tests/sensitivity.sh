#!/bin/bash
# 参数敏感性扫描：k / band / seed-step 对召回率与耗时的影响
set -euo pipefail
ROOT="/mnt/c/Users/31440/AppData/Roaming/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a9a404a7ef71d3d93b68e28/proj_seqalign"
cd "$ROOT"
BIN=./build/seqalign
REF=tests/tmp/large.fa
RD=tests/tmp/large.fq
TR=tests/tmp/large.truth

printf "%-6s %-6s %-6s %10s %10s %12s %12s\n" "k" "band" "step" "候选对" "起点一致" "比对ms" "召回"
for cfg in "11 8 8" "13 8 8" "15 4 8" "15 8 4" "15 8 8" "15 8 16" "15 16 8"; do
  set -- $cfg
  k=$1; band=$2; step=$3
  out=$($BIN gpu $REF $RD tests/tmp/sens.txt --k $k --band $band --seed-step $step --truth $TR 2>&1)
  cand=$(echo "$out" | sed -n 's/.*候选对 = \([0-9]*\).*/\1/p')
  al=$(echo "$out" | grep "有来源 read" | sed -n 's/.*被比对 (\([0-9.]*\)%).*起点完全一致 \([0-9]*\).*/\2/p')
  tot=$(echo "$out" | grep "有来源 read" | sed -n 's/.*起点完全一致 [0-9]* (\([0-9.]*\)%).*/\1/p')
  unk=$(echo "$out" | grep "随机 read" | sed -n 's/.*unknown_origin (\([0-9.]*\)%).*/\1/p')
  ms=$(echo "$out" | sed -n 's/.*T_align  = \([0-9.]*\) ms/\1/p')
  printf "%-6s %-6s %-6s %10s %10s %12s  起点%s%% 随机%s%%\n" "$k" "$band" "$step" "$cand" "$al" "$ms" "$tot" "$unk"
done
