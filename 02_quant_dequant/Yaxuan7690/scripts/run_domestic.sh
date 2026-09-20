#!/usr/bin/env bash
# =============================================================================
#  run_domestic.sh -- 在国产 GPU 平台上把本项目跑通并留下证据
#
#  用法:
#      bash scripts/run_domestic.sh nvidia
#      bash scripts/run_domestic.sh iluvatar
#      bash scripts/run_domestic.sh moore
#      bash scripts/run_domestic.sh metax
#
#  行为:
#      1. 检查编译器、SDK 环境变量、CUDA 兼容层与需不需要额外开关
#      2. (moore) 自动探出可用的 --offload-arch 代号
#      3. make -n 干跑一遍, Makefile 参数有问题在这一步就暴露
#      4. 编译主程序与三个测试
#      5. 先跑移植探针 (test_portability), 它会逐项报告平台缺什么能力
#      6. 跑两个正确性测试套件
#      7. 跑一个小规模的端到端实验, 产生误差/压缩率/带宽日志
#      8. 编一个带宽对照 kernel 并量出这台机器的纯搬运上限(达成率的分母)
#      9. 全过程 tee 到 data/out/domestic_<platform>.log
#
#  设计成不会因为某一步失败就退出: 租来的机器按小时计费, 一次跑完拿到尽可能多
#  的信息比中途停住更有价值.
#
#  如果探针报告某个能力缺失, 用 LPQ_EXTRA 传入对应开关, 例如:
#      LPQ_EXTRA="-DLPQ_NO_ATOMIC_ADD_F64=1" bash scripts/run_domestic.sh moore
#
#  摩尔线程 MUSA 的一个额外坑: 它的运行时对象叫 musaXxx, 而本项目源码写的是
#  cudaXxx. 官方给了三条现成的路消化这个差异(mcc 的 -cuda_wrapper 兼容封装模式 /
#  SDK >= 5.2.0 的 MUSA Mapping 插件 / musify 离线迁移), 都接在 Makefile 的
#  moore 分支里. 本脚本负责先探清这台机器有哪条可用, 并让架构探测用同一套参数,
#  免得在按小时计费的机器上盲试.
# =============================================================================
set -u

PLATFORM="${1:-}"
if [ -z "${PLATFORM}" ]; then
    echo "usage: bash scripts/run_domestic.sh <nvidia|iluvatar|moore|metax>"
    exit 2
fi

case "${PLATFORM}" in
    nvidia|iluvatar|moore|metax) ;;
    *) echo "unknown platform '${PLATFORM}'"; exit 2 ;;
esac

# 下面几个只对 moore 有意义. 先给默认值: 脚本开着 set -u, 引用未定义变量会直接退出
MUSA_ARCH="${MUSA_ARCH:-}"
MUSA_ARCH_CANDIDATES="${MUSA_ARCH_CANDIDATES:-mp_21 mp_22 mp_10 mp_31 mp_20 mp_30}"
MUSA_PLUGIN="${MUSA_PLUGIN:-}"
MUSA_CUDA2MUSA="${MUSA_CUDA2MUSA:-}"

# 切到仓库根目录 (本脚本在 scripts/ 下)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || exit 1

case "${PLATFORM}" in
    nvidia)   CC_BIN="${CC_BIN:-nvcc}" ;;
    iluvatar) CC_BIN="${CC_BIN:-clang++}" ;;
    moore)    CC_BIN="${CC_BIN:-mcc}" ;;
    metax)    CC_BIN="${CC_BIN:-mxcc}" ;;
esac

mkdir -p data/out data/tensors
LOG="data/out/domestic_${PLATFORM}.log"
: > "${LOG}"

# 同时输出到屏幕和日志
run_logged() {
    "$@" 2>&1 | tee -a "${LOG}"
    return "${PIPESTATUS[0]}"
}

{
    echo "======================================================================"
    echo " platform      : ${PLATFORM}"
    echo " compiler      : ${CC_BIN}"
    echo " date          : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo " host          : $(uname -srm)"
    echo " LPQ_EXTRA     : ${LPQ_EXTRA:-<none>}"
    echo "======================================================================"
} | tee -a "${LOG}"

# ---------------------------------------------------------------------------
# 0. 环境
# ---------------------------------------------------------------------------
echo "" | tee -a "${LOG}"
echo "---- [0/5] environment ----" | tee -a "${LOG}"

# 国产平台的 SDK 环境变量必须**先**设好再探测编译器: 否则 SDK 没进 PATH 时,
# 下面那条 command -v 会把"环境变量没配"误报成"工具链没装".
case "${PLATFORM}" in
    moore)
        MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
        echo "MUSA_HOME     : ${MUSA_HOME}" | tee -a "${LOG}"
        if [ -d "${MUSA_HOME}" ]; then
            export MUSA_HOME
            export PATH="${MUSA_HOME}/bin:${PATH}"
            export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${LD_LIBRARY_PATH:-}"
            echo "musa bin      : $(ls "${MUSA_HOME}/bin" 2>/dev/null | tr '\n' ' ')" | tee -a "${LOG}"
        else
            echo "!! ${MUSA_HOME} does not exist; set MUSA_HOME=..." | tee -a "${LOG}"
        fi
        # 编译器与 SDK 版本: 上机后第一批要看的东西
        mcc --version 2>&1 | head -5 | tee -a "${LOG}"
        if command -v musa_version_query >/dev/null 2>&1; then
            musa_version_query 2>&1 | tee -a "${LOG}"
        fi
        # 本项目源码写的是 cudaXxx, 而 MUSA 运行时叫 musaXxx. 官方的三条路都接在
        # Makefile 里了, 这里只负责把探测结果讲清楚 —— make 会按同样的依据自动选:
        #   1. libcuda2musa.so 在       -> mcc -mtgpu -cuda_wrapper -lcuda2musa
        #   2. 只有 Mapping 插件(>=5.2) -> mcc -x musa -fplugin=<插件>
        #   3. 都没有                   -> 先用 musify 迁移源码, 再 MUSA_MODE=plain
        MUSA_PLUGIN=""
        MUSAMAPPING="${MUSA_HOME}/tools/musaMapping"
        MUSA_CUDA2MUSA=""
        for d in "${MUSA_HOME}/lib" "${MUSA_HOME}/lib64"; do
            if [ -f "${d}/libcuda2musa.so" ]; then
                MUSA_CUDA2MUSA="${d}"
                break
            fi
        done
        if [ -n "${MUSA_CUDA2MUSA}" ]; then
            echo "CUDA 兼容层   : libcuda2musa.so 在 ${MUSA_CUDA2MUSA}" | tee -a "${LOG}"
            echo "                -> make 用 -mtgpu -cuda_wrapper -lcuda2musa, 源码不用改" | tee -a "${LOG}"
        elif [ -f "${MUSAMAPPING}/libMusaMapping.so" ]; then
            MUSA_PLUGIN="${MUSAMAPPING}/libMusaMapping.so"
            echo "CUDA 兼容层   : 只有 MUSA Mapping 插件 (SDK >= 5.2.0)" | tee -a "${LOG}"
            echo "                -> make 用 -x musa -fplugin=<插件> -lmusart -lmusa" | tee -a "${LOG}"
        else
            echo "CUDA 兼容层   : 既没有 libcuda2musa.so 也没有 MUSA Mapping 插件" | tee -a "${LOG}"
            echo "                -> 先用官方工具迁移源码, 再按 plain 模式编译:" | tee -a "${LOG}"
            echo "                   musify-text --inplace -- src/*.cu tests/*.cu" | tee -a "${LOG}"
            echo "                   (musify 需要 python3 与 pip install ahocorapy)" | tee -a "${LOG}"
            echo "                   make PLATFORM=moore MUSA_MODE=plain build" | tee -a "${LOG}"
        fi
        if command -v musify-text >/dev/null 2>&1; then
            echo "musify-text   : $(command -v musify-text)" | tee -a "${LOG}"
        fi
        ;;
    iluvatar)
        CORE_X_HOME="${CORE_X_HOME:-/usr/local/corex}"
        echo "CORE_X_HOME   : ${CORE_X_HOME}" | tee -a "${LOG}"
        if [ -d "${CORE_X_HOME}" ]; then
            export LD_LIBRARY_PATH="${CORE_X_HOME}/lib64:${LD_LIBRARY_PATH:-}"
        else
            echo "!! ${CORE_X_HOME} does not exist; set CORE_X_HOME=..." | tee -a "${LOG}"
        fi
        ;;
    metax)
        MACA_HOME="${MACA_HOME:-/opt/maca}"
        echo "MACA_HOME     : ${MACA_HOME}" | tee -a "${LOG}"
        if [ -d "${MACA_HOME}" ]; then
            export LD_LIBRARY_PATH="${MACA_HOME}/lib:${LD_LIBRARY_PATH:-}"
        else
            echo "!! ${MACA_HOME} does not exist (this may be fine if mxcc is self-contained)" \
                | tee -a "${LOG}"
        fi
        ;;
esac

if ! command -v "${CC_BIN}" >/dev/null 2>&1; then
    echo "!! '${CC_BIN}' not found in PATH. Set CC_BIN=... or fix the toolchain." | tee -a "${LOG}"
    exit 1
fi
echo "compiler path : $(command -v "${CC_BIN}")" | tee -a "${LOG}"

# 各家的设备监控工具(相当于 nvidia-smi): 有哪个就打哪个, 一个都没有也不影响
for gmi_tool in nvidia-smi mthreads-gmi mx-smi ixsmi; do
    if command -v "${gmi_tool}" >/dev/null 2>&1; then
        echo "" | tee -a "${LOG}"
        "${gmi_tool}" 2>&1 | tee -a "${LOG}"
    fi
done

# ---------------------------------------------------------------------------
# 0.5 (仅 moore) 目标架构代号探测
#   MUSA 每个 GPU 架构一个代号(mp_21 / mp_22 / mp_31 ...), mcc 的 --offload-arch
#   用的就是它. 填错的后果是编不过、或者编过了产物在卡上起不来, 而租来的机器上
#   并没有一张"卡型号 -> 代号"的对照表可查 —— 所以直接试: 用与正式构建完全相同
#   的一组参数, 逐个候选编一个小 kernel 并真的跑一次, 第一个既编得过、又跑得出
#   正确答案的候选就是答案.
#   探测源码刻意写成 MUSA 原生写法(musaMalloc / musa_runtime.h): 无论开不开
#   MUSA Mapping 插件它都编得过, 于是这只测"架构对不对"这一件事.
#
#   探测 kernel 必须**真的用到低精度 intrinsic**: MUSA 头文件里 __bfloat16_as_ushort
#   / __ushort_as_bfloat16 带 __MUSA_ARCH__ >= 220 的守卫, 而项目设备侧本来就依赖
#   它们(vec_io.cuh 的 bf16 编解码走的是按位互转). 只写一句 o[0] = 42 的探针会认为
#   mcc 的默认输出可用(它同时编 mp_21 + mp_22), 直到正式构建才报 undeclared
#   identifier —— 在按小时计费的机器上白跑一轮. MTT S4000 上实测:
#       default -> 编译失败   mp_21 -> 编译失败   mp_10 -> 编译失败
#       mp_22   -> 编过且跑出 42 (正确)          mp_31 -> 编过但起不来, 被运行检查拒掉
# ---------------------------------------------------------------------------
if [ "${PLATFORM}" = "moore" ]; then
    echo "" | tee -a "${LOG}"
    echo "---- [0.5/5] MUSA arch probe ----" | tee -a "${LOG}"
    if [ -n "${MUSA_ARCH}" ]; then
        echo "MUSA_ARCH     : '${MUSA_ARCH}' (来自环境变量, 跳过探测)" | tee -a "${LOG}"
    else
        ARCH_DIR="data/out/_archprobe"
        mkdir -p "${ARCH_DIR}"
        cat > "${ARCH_DIR}/arch_probe.cu" <<'PROBE_EOF'
#include <musa_runtime.h>
#include <musa_bf16.h>
#include <cstdio>
// 用上 bf16 的按位互转: 它们在 MUSA 头文件里有 __MUSA_ARCH__ >= 220 的守卫, 于是
// "只编 mp_21" 或 "默认同时编 mp_21 + mp_22" 的产物都编不过, 而项目源码在设备侧
// 依赖这三个 intrinsic. 详见本段开头的说明.
__global__ void k_write(int* o) {
    const __mt_bfloat16 b = __float2bfloat16_rn(1.0f);
    const unsigned short u = __bfloat16_as_ushort(b);
    o[0] = (__ushort_as_bfloat16(u) == b) ? 42 : 7;
}
int main(void) {
    int* d = 0;
    int h = 0;
    if (musaMalloc((void**)&d, sizeof(int)) != musaSuccess) { std::printf("alloc-fail"); return 2; }
    k_write<<<1, 1>>>(d);
    if (musaGetLastError() != musaSuccess) { std::printf("launch-fail"); return 3; }
    musaMemcpy(&h, d, sizeof(int), musaMemcpyDeviceToHost);
    musaFree(d);
    std::printf("%d", h);
    return h == 42 ? 0 : 4;
}
PROBE_EOF
        ARCH_OK=0
        # 探测用的编译/链接参数与 Makefile 的 auto 模式保持同一套判定, 否则探出来的
        # 代号可能只对探测有效、对正式构建无效
        PROBE_MODE="-mtgpu"
        PROBE_LIB="-L${MUSA_HOME}/lib -lmusart"
        if [ -n "${MUSA_CUDA2MUSA}" ]; then
            PROBE_MODE="-mtgpu -cuda_wrapper"
            PROBE_LIB="-L${MUSA_CUDA2MUSA} -lcuda2musa"
        elif [ -n "${MUSA_PLUGIN}" ]; then
            PROBE_MODE="-x musa -I. -fplugin=${MUSA_PLUGIN}"
            PROBE_LIB="-L${MUSA_HOME}/lib -Wl,--no-as-needed -lmusart -lmusa"
        fi
        echo "探测参数      : ${PROBE_MODE} ${PROBE_LIB}" | tee -a "${LOG}"
        # 第一个候选用空串: 先看 mcc 自己决定的行不行, 再按已知代号从新到旧试
        # (MUSA 3.1.0 的有效代号是 mp_10 / mp_21 / mp_22; 不传则默认出前两个)
        for cand in "" ${MUSA_ARCH_CANDIDATES}; do
            label="${cand:-default}"
            arch_flag=""
            [ -n "${cand}" ] && arch_flag="--offload-arch=${cand}"
            if MUSAMAPPING_PATH="${MUSAMAPPING}" MUSA_INCLUDE_PATH="${MUSA_HOME}/include" \
               "${CC_BIN}" ${PROBE_MODE} ${arch_flag} -I"${MUSA_HOME}/include" \
                   "${ARCH_DIR}/arch_probe.cu" ${PROBE_LIB} \
                   -o "${ARCH_DIR}/probe_${label}" > "${ARCH_DIR}/probe_${label}.log" 2>&1; then
                got="$("${ARCH_DIR}/probe_${label}" 2>&1 | tail -1)"
                if [ "${got}" = "42" ]; then
                    echo "MUSA_ARCH     : '${cand}' 可用 (编得过, 并在设备上跑出 42)" | tee -a "${LOG}"
                    MUSA_ARCH="${cand}"
                    ARCH_OK=1
                    break
                fi
                echo "  候选 '${label}': 编过了但运行返回 '${got}'" | tee -a "${LOG}"
            else
                echo "  候选 '${label}': 编译失败 -> ${ARCH_DIR}/probe_${label}.log" | tee -a "${LOG}"
            fi
        done
        if [ "${ARCH_OK}" = "0" ]; then
            echo "MUSA_ARCH     : 所有候选都没成功, 留空交给 mcc (正式构建很可能失败)" | tee -a "${LOG}"
            echo "                日志在 ${ARCH_DIR}/*.log; 也可以直接指定代号重跑:" | tee -a "${LOG}"
            echo "                MUSA_ARCH=mp_xx bash scripts/run_domestic.sh moore" | tee -a "${LOG}"
        fi
        export MUSA_ARCH
    fi
fi

# ---------------------------------------------------------------------------
# 0.9 干跑一遍 (make -n): 只打印命令不真编译, Makefile 的分支与参数有问题会在
#     这一步暴露 —— 比让编译器报一堆错更快, 也省租机时间
# ---------------------------------------------------------------------------
echo "" | tee -a "${LOG}"
echo "---- [0.9/5] dry run (make -n) ----" | tee -a "${LOG}"
DRY_OUT="data/out/_dryrun.txt"
if make -n PLATFORM="${PLATFORM}" build > "${DRY_OUT}" 2>&1; then
    head -6 "${DRY_OUT}" | tee -a "${LOG}"
else
    echo "!! make -n 失败, 先看这里:" | tee -a "${LOG}"
    head -20 "${DRY_OUT}" | tee -a "${LOG}"
fi

# ---------------------------------------------------------------------------
# 1. 编译
# ---------------------------------------------------------------------------
echo "" | tee -a "${LOG}"
echo "---- [1/5] build ----" | tee -a "${LOG}"
run_logged make PLATFORM="${PLATFORM}" build
BUILD_RC=$?
echo "build exit code: ${BUILD_RC}" | tee -a "${LOG}"

# ---------------------------------------------------------------------------
# 2. 移植探针 (先跑, 它最能说明平台缺什么)
# ---------------------------------------------------------------------------
echo "" | tee -a "${LOG}"
echo "---- [2/5] portability probe ----" | tee -a "${LOG}"
if [ -x build/test_portability ]; then
    run_logged ./build/test_portability
    echo "probe exit code: $?" | tee -a "${LOG}"
else
    echo "!! build/test_portability was not produced - the failure is at compile time," | tee -a "${LOG}"
    echo "   look at the errors above. See the header of tests/test_portability.cu for"  | tee -a "${LOG}"
    echo "   the checklist of switches that map to each error."                      | tee -a "${LOG}"
fi

# ---------------------------------------------------------------------------
# 3. 正确性测试
# ---------------------------------------------------------------------------
echo "" | tee -a "${LOG}"
echo "---- [3/5] correctness suites ----" | tee -a "${LOG}"
for t in test_formats test_pipeline; do
    if [ -x "build/${t}" ]; then
        echo ">>> ${t}" | tee -a "${LOG}"
        run_logged "./build/${t}"
        echo "${t} exit code: $?" | tee -a "${LOG}"
    else
        echo "!! build/${t} missing" | tee -a "${LOG}"
    fi
done

# ---------------------------------------------------------------------------
# 4. 端到端实验 (误差 / 压缩率 / 带宽)
# ---------------------------------------------------------------------------
echo "" | tee -a "${LOG}"
echo "---- [4/5] end-to-end experiment ----" | tee -a "${LOG}"

# 带宽对照工具: 它给出这台机器的纯搬运上限. 换一台卡就要重测一次, 因为跨机器
# 比较 GB/s 没有意义, 只有"实测 / 本机上限"的达成率才有意义
echo ">>> bw_ceiling" | tee -a "${LOG}"
run_logged make PLATFORM="${PLATFORM}" bw
if [ -x build/bw_ceiling ]; then
    run_logged ./build/bw_ceiling
else
    echo "!! build/bw_ceiling 没编出来 (不影响下面的端到端实验)" | tee -a "${LOG}"
fi
if [ -x build/lpq_quant ] && command -v python3 >/dev/null 2>&1; then
    if [ ! -f data/tensors/normal_512.tensor ]; then
        run_logged python3 tools/gen_data.py --out-dir data/tensors --name normal_512 \
            --rows 512 --cols 512 --dist normal --dtype fp32
    fi
    for cfg in mxfp8-e4m3_block_fp16_nearest mxfp8-e5m2_block_fp16_nearest \
               nvfp4_block_fp16_nearest; do
        if [ -f "data/params/${cfg}.params" ]; then
            echo ">>> ${cfg}" | tee -a "${LOG}"
            run_logged ./build/lpq_quant --input data/tensors/normal_512.tensor \
                --params "data/params/${cfg}.params" \
                --out-dir "data/out" --cpu-check
        fi
    done
else
    echo "skipped (需要 build/lpq_quant 与 python3)" | tee -a "${LOG}"
fi

# ---------------------------------------------------------------------------
# 5. 收尾
# ---------------------------------------------------------------------------
echo "" | tee -a "${LOG}"
echo "---- [5/5] done ----" | tee -a "${LOG}"
echo "log written to ${LOG}" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"
echo "Next: copy ${LOG} back and note in the report which platform worked," | tee -a "${LOG}"
echo "      plus any -DLPQ_* switch that had to be passed via LPQ_EXTRA." | tee -a "${LOG}"
exit 0
