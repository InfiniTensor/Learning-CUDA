#!/usr/bin/env bash
# validate.sh —— 正确性门禁。输出 PASS/FAIL 表，任一 FAIL 则退出码非零。
#
# 8 段共 16 项。段号固定：1 二进制布局 / 2 生成初值 / 3 二体圆轨道 /
# 4 二体偏心轨道 / 5 Euler vs Leapfrog / 6 多体守恒（Plummer）/ 7 GPU 诊断量 /
# 8 CSV 输出。具体项数与 PASS 数由脚本末尾自己打印，文档里引用那个数字。
# 依赖：binutils/od/dd/cmp（coreutils），不需要 python、不需要任何测试框架。
# 注意：PASS 数是"本次运行"的结果，不要在文档里把它当成硬件无关的常数。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build"
OUT="${ROOT}/results/validate"
mkdir -p "${OUT}" "${ROOT}/data"

fail=0
pass_count=0

report() {
  # $1 = 名称, $2 = 0/1 是否通过, $3 = 备注
  if [ "$2" -eq 0 ]; then
    printf '  %-42s PASS  %s\n' "$1" "${3:-}"
    pass_count=$((pass_count + 1))
  else
    printf '  %-42s FAIL  %s\n' "$1" "${3:-}"
    fail=$((fail + 1))
  fi
}

echo "=== 1. particle-major binary layout ==="
# 不依赖任何单测二进制：用两轮写出做一个字节级交叉判定。
#   轮 A：record_interval 推到极大、num_steps=1 -> 只落 1 帧。此时
#          [粒子][记录] 与 [记录][粒子] 两种展平完全等价，A 的载荷
#          就是初值的权威字节布局。
#   轮 B：正常多帧。断言 B 中"每个粒子的第 0 帧"拼起来 == A 的载荷。
# 若写出侧把记录主序缓冲直接 dump（即漏了转置），B 的第 0 帧列会与
# 帧内粒子顺序错位，cmp 立即失败——而这正是那种"文件大小 / P / R
# 全对、程序不报错、只有动画错乱"的 bug（见 trajectory_io.cpp）。
"${BIN}/nbody_ic" --preset two_body --out "${OUT}/layout_ic.txt"
sed -e 's/^num_steps.*/num_steps = 1/' \
    -e 's/^record_interval.*/record_interval = 1000000000/' \
    "${ROOT}/data/params_two_body.txt" > "${OUT}/params_layout_a.txt"
"${BIN}/nbody_sim" --bodies "${OUT}/layout_ic.txt" \
  --params "${OUT}/params_layout_a.txt" \
  --out "${OUT}/layout_a.bin" > "${OUT}/io_roundtrip.log" 2>&1
"${BIN}/nbody_sim" --bodies "${OUT}/layout_ic.txt" \
  --params "${ROOT}/data/params_two_body.txt" \
  --out "${OUT}/layout_b.bin" >> "${OUT}/io_roundtrip.log" 2>&1
read -r LP LR < <(od -An -td4 -N8 "${OUT}/layout_b.bin" 2>/dev/null)
layout_ok=1
if [ -n "${LP:-}" ] && [ -n "${LR:-}" ] && [ "${LP:-0}" -ge 2 ] && [ "${LR:-0}" -ge 2 ]; then
  sa=$(stat -c%s "${OUT}/layout_a.bin" 2>/dev/null || echo 0)
  sb=$(stat -c%s "${OUT}/layout_b.bin" 2>/dev/null || echo 0)
  # 尺寸闸门：顺带把"dd 失败 -> 两个文件都空 -> cmp 误判相等"堵死
  if [ "${sa}" -eq $((8 + LP * 12)) ] && [ "${sb}" -eq $((8 + LP * LR * 12)) ]; then
    : > "${OUT}/layout_expected.bin"
    for ((ip = 0; ip < LP; ++ip)); do
      dd if="${OUT}/layout_b.bin" bs=1 skip=$((8 + ip * LR * 12)) count=12 \
        >> "${OUT}/layout_expected.bin" 2>/dev/null
    done
    dd if="${OUT}/layout_a.bin" bs=1 skip=8 count=$((LP * 12)) \
      > "${OUT}/layout_ic.bin" 2>/dev/null
    if cmp -s "${OUT}/layout_expected.bin" "${OUT}/layout_ic.bin"; then
      layout_ok=0
    fi
  fi
fi
if [ "${layout_ok}" -ne 0 ]; then
  report "particle-major binary layout" 1 "see ${OUT}/io_roundtrip.log"
  cat "${OUT}/io_roundtrip.log"
  printf '    P=%s R=%s expected=%s ic=%s\n' "${LP:-?}" "${LR:-?}" \
    "$(stat -c%s "${OUT}/layout_expected.bin" 2>/dev/null || echo -)" \
    "$(stat -c%s "${OUT}/layout_ic.bin" 2>/dev/null || echo -)"
else
  report "particle-major binary layout" 0 \
    "P=${LP} R=${LR}, $((${LP} * 12)) bytes cross-checked"
fi

echo
echo "=== 2. generate initial conditions ==="
"${BIN}/nbody_ic" --preset two_body     --out "${ROOT}/data/two_body.txt"
"${BIN}/nbody_ic" --preset two_body_ecc --out "${ROOT}/data/two_body_ecc.txt"
"${BIN}/nbody_ic" --preset plummer -n 4096 --out "${ROOT}/data/plummer_4096.txt"
"${BIN}/nbody_ic" --preset uniform -n 4096 --out "${ROOT}/data/uniform_4096.txt"

echo
echo "=== 3. two-body circular orbit (10 periods, leapfrog) ==="
# 期望：a≈1（相对间距的半长轴）、e≈0、|dE/E0| 有界且很小。
# --diag-interval 1000：10 个周期内留 45 个诊断点，供
# tools/plot_energy.py 画能量/角动量漂移曲线（N=2 的诊断代价可忽略）。
"${BIN}/nbody_sim" \
  --bodies "${ROOT}/data/two_body.txt" \
  --params "${ROOT}/data/params_two_body.txt" \
  --out "${OUT}/two_body_leapfrog.bin" \
  --log "${OUT}/two_body_leapfrog.json" \
  --diag-interval 1000 \
  --cpu-check > "${OUT}/two_body_leapfrog.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  report "two-body leapfrog run" 1 "exit=$rc"
  tail -20 "${OUT}/two_body_leapfrog.log"
else
  report "two-body leapfrog run" 0
  grep -E "^two-body:|dE/E0|gpu-vs-cpu" "${OUT}/two_body_leapfrog.log" | sed 's/^/    /'

  # 能量误差判据：leapfrog 是辛的，10 个周期后相对误差应远小于 1e-3。
  err=$(grep -oP '\|dE/E0\|=\K[0-9.e+-]+' "${OUT}/two_body_leapfrog.log" | head -1)
  awk -v e="${err}" 'BEGIN { exit (e < 1e-3) ? 0 : 1 }'
  report "leapfrog energy error < 1e-3" $? "|dE/E0|=${err}"

  # 轨道要素：半长轴应保持 ~1，偏心率应保持 ~0（圆轨道）。
  a=$(grep -oP '^two-body: a=\K[0-9.e+-]+' "${OUT}/two_body_leapfrog.log" | head -1)
  ecc=$(grep -oP ' e=\K[0-9.e+-]+' "${OUT}/two_body_leapfrog.log" | head -1)
  awk -v a="${a}" 'BEGIN { d = a - 1.0; if (d < 0) d = -d; exit (d < 0.01) ? 0 : 1 }'
  report "semi-major axis within 1% of 1.0" $? "a=${a}"
  awk -v e="${ecc}" 'BEGIN { exit (e < 0.01) ? 0 : 1 }'
  report "eccentricity stays circular (e<0.01)" $? "e=${ecc}"

  # 轨道周期 vs 解析值。等质量双星（m=1 each、G=1、相对间距 d=1）的
  # 解析周期是 T = 2*pi*sqrt(a^3/mu) = 2*pi/sqrt(2) = 4.442883。
  # 这一项不是半长轴漂移的同义重复：a 由瞬时状态推出，而 period 是
  # 开普勒第三定律给出的整体量，跑满 10 个周期后它对"轨道被积分器拧歪"
  # 更敏感（a 的漂移可能互相抵消，周期不会）。判据直接取自需求文档的
  # "周期与解析值之差 < 1%"。
  per=$(grep -oP 'period=\K[0-9.e+-]+' "${OUT}/two_body_leapfrog.log" | head -1)
  awk -v p="${per}" 'BEGIN { d = p - 4.442883; if (d < 0) d = -d;
                             exit (d / 4.442883 < 0.01) ? 0 : 1 }'
  report "orbital period within 1% of analytic" $? "T=${per} (analytic 4.442883)"

  # GPU 与 CPU 的加速度逐元素比对。这是本版的主力数值门禁：
  # 力计算只有一份实现，参照物必须是独立的 CPU 实现（见 main.cu 的说明）。
  if grep -q "gpu-vs-cpu acc:.*PASS" "${OUT}/two_body_leapfrog.log"; then
    report "gpu-vs-cpu acceleration match" 0
  else
    report "gpu-vs-cpu acceleration match" 1
  fi
fi

echo
echo "=== 4. two-body eccentric orbit ==="
# 0.8x 圆轨道速度 → e = 1 - 0.8^2 = 0.36
"${BIN}/nbody_sim" \
  --bodies "${ROOT}/data/two_body_ecc.txt" \
  --params "${ROOT}/data/params_two_body.txt" \
  --out "${OUT}/two_body_ecc.bin" \
  --log "${OUT}/two_body_ecc.json" > "${OUT}/two_body_ecc.log" 2>&1
if [ $? -ne 0 ]; then
  report "eccentric orbit run" 1
  tail -20 "${OUT}/two_body_ecc.log"
else
  report "eccentric orbit run" 0
  grep -E "^two-body:" "${OUT}/two_body_ecc.log" | sed 's/^/    /'
  ecc=$(grep -oP ' e=\K[0-9.e+-]+' "${OUT}/two_body_ecc.log" | head -1)
  awk -v e="${ecc}" 'BEGIN { d = e - 0.36; if (d < 0) d = -d; exit (d < 0.02) ? 0 : 1 }'
  report "eccentricity matches analytic 0.36" $? "e=${ecc}"
fi

echo
echo "=== 5. euler vs leapfrog energy behaviour ==="
# Euler 一阶非辛 → 能量单调漂移，误差应显著大于 leapfrog。
# 这一项不设 PASS 判据（Euler 本来就该差），只记录数字供报告作图。
# --diag-interval 1000 与 §3 保持一致，两份日志才能画在同一张图上
# （tools/plot_energy.py 的两条曲线）。
"${BIN}/nbody_sim" \
  --bodies "${ROOT}/data/two_body.txt" \
  --params "${ROOT}/data/params_two_body_euler.txt" \
  --out "${OUT}/two_body_euler.bin" \
  --log "${OUT}/two_body_euler.json" \
  --diag-interval 1000 > "${OUT}/two_body_euler.log" 2>&1
err_euler=$(grep -oP '\|dE/E0\|=\K[0-9.e+-]+' "${OUT}/two_body_euler.log" | head -1)
err_lf=$(grep -oP '\|dE/E0\|=\K[0-9.e+-]+' "${OUT}/two_body_leapfrog.log" | head -1)
printf '    euler    |dE/E0| = %s\n' "${err_euler:-n/a}"
printf '    leapfrog |dE/E0| = %s\n' "${err_lf:-n/a}"
if [ -n "${err_euler}" ] && [ -n "${err_lf}" ]; then
  awk -v a="${err_euler}" -v b="${err_lf}" 'BEGIN { exit (a > b) ? 0 : 1 }'
  report "leapfrog beats euler on energy" $? "ratio=$(awk -v a="${err_euler}" -v b="${err_lf}" 'BEGIN{printf "%.1fx", a/b}')"
fi

echo
echo "=== 6. many-body conservation (Plummer N=4096) ==="
"${BIN}/nbody_sim" \
  --bodies "${ROOT}/data/plummer_4096.txt" \
  --params "${ROOT}/data/params_cluster.txt" \
  --out "${OUT}/plummer_4096.bin" \
  --log "${OUT}/plummer_4096.json" \
  --diag-interval 500 --cpu-check > "${OUT}/plummer_4096.log" 2>&1
if [ $? -ne 0 ]; then
  report "plummer run" 1
  tail -20 "${OUT}/plummer_4096.log"
else
  report "plummer run" 0
  grep -E "^  step|dE/E0|momentum|angular|gpu-vs-cpu" "${OUT}/plummer_4096.log" | sed 's/^/    /'
  err=$(grep -oP '\|dE/E0\|=\K[0-9.e+-]+' "${OUT}/plummer_4096.log" | head -1)
  # 多体 + FP32 + 近距离交会，1e-2 是合理范围。
  awk -v e="${err}" 'BEGIN { exit (e < 1e-2) ? 0 : 1 }'
  report "many-body energy error < 1e-2" $? "|dE/E0|=${err}"

  # 角动量漂移。有心力下 |L| 严格守恒，漂移只来自浮点舍入与力的
  # 反对称性缺失（每个目标粒子独立求和，F_ij != -F_ji）。
  # 判据取与能量同一量级（1e-2）：两者对力误差的响应不同，
  # 都过才说明力计算本身是对的，而不只是总能量碰巧对上了。
  lerr=$(grep -oP '^angular .*rel=\K[0-9.e+-]+' "${OUT}/plummer_4096.log" | head -1)
  awk -v e="${lerr}" 'BEGIN { exit (e < 1e-2) ? 0 : 1 }'
  report "many-body angular momentum drift < 1e-2" $? "rel=${lerr}"
  if grep -q "gpu-vs-cpu acc:.*PASS" "${OUT}/plummer_4096.log"; then
    report "gpu-vs-cpu acceleration match (N=4096)" 0
  else
    report "gpu-vs-cpu acceleration match (N=4096)" 1
  fi
fi

echo
echo "=== 7. gpu diagnostics vs host reference ==="
# 诊断量是裁定"FP32 主线够不够准"的尺子，尺子本身错了所有能量结论都不可信。
# GPU 版与主机版数学相同、求和顺序不同，差异应在 double 浮点重排级别（1e-10）。
# --check-diag 在任一指标超差时会以非零退出。
"${BIN}/nbody_sim" \
  --bodies "${ROOT}/data/plummer_4096.txt" \
  --params "${ROOT}/data/params_cluster.txt" \
  --out /dev/null \
  --check-diag > "${OUT}/check_diag.log" 2>&1
if [ $? -eq 0 ]; then
  report "gpu diagnostics match host reference" 0
  grep -E "rel=" "${OUT}/check_diag.log" | sed 's/^/    /'
else
  report "gpu diagnostics match host reference" 1 "see ${OUT}/check_diag.log"
  grep -E "rel=" "${OUT}/check_diag.log" | sed 's/^/    /'
fi

echo
echo "=== 8. csv format works ==="
sed 's/^integrator.*/integrator = "leapfrog"/' "${ROOT}/data/params_bench.txt" \
  > "${OUT}/params_csv.txt"
echo 'format = "csv"' >> "${OUT}/params_csv.txt"
"${BIN}/nbody_sim" \
  --bodies "${ROOT}/data/uniform_4096.txt" \
  --params "${OUT}/params_csv.txt" \
  --out "${OUT}/uniform_4096.csv" > "${OUT}/csv.log" 2>&1
report "csv output" $?
if [ -f "${OUT}/uniform_4096.csv" ]; then
  csv_size=$(stat -c%s "${OUT}/uniform_4096.csv" 2>/dev/null || echo 0)
  printf '    csv %s bytes (for the report: csv is ~6-8x larger than binary)\n' "${csv_size}"
fi

echo
echo "======================================"
printf 'validation: %d passed, %d failed\n' "${pass_count}" "${fail}"
echo "logs and json in ${OUT}/"
exit $((fail > 0 ? 1 : 0))
