# =============================================================================
#  Makefile -- MXFP8 / NVFP4 低精度格式软件模拟与反量化 (选题二)
#
#  用法:
#    make                      # 编译并运行全部测试 (默认)
#    make build                # 只编译
#    make test                 # 只运行测试
#    make probe                # 只运行国产平台移植探针 (换平台后先跑这个)
#    make app                  # 只编译主程序 build/lpq_quant
#    make PLATFORM=iluvatar    # 切换国产平台 (nvidia / iluvatar / moore / metax)
#    make bw                   # 只编译带宽天花板对照 kernel build/bw_ceiling
#    make clean                # 清理
#
#  说明: 本程序只使用基础 CUDA C++ (无 FP8/FP4 Tensor Core、无 cuda_fp8/cuda_fp4
#  头文件、无 Hopper/Blackwell 语义), 因此切换平台只需要换编译器与架构标志.
#  六处源码级差异集中在 include/lpq/platform.hpp, 若某个平台缺某种能力, 用
#  LPQ_EXTRA 追加开关即可, 例如:
#      make PLATFORM=moore LPQ_EXTRA="-DLPQ_NO_ATOMIC_ADD_F64=1"
#
#  例外是摩尔线程 MUSA: 它的运行时叫 musaXxx 而不是 cudaXxx (本项目有 200 多处
#  cudaXxx 调用点), 这处构建级差异由 SDK 自带的编译期兼容层 MUSA Mapping 消化,
#  源码保持 CUDA 写法不变 —— 见下面 moore 分支的长注释.
# =============================================================================

PLATFORM    ?= nvidia
BUILD_DIR   := build
BIN         := $(BUILD_DIR)

# 追加给编译器的自定义开关 (国产平台缺少某项能力时使用)
LPQ_EXTRA   ?=

# --------------------------- 摩尔线程 MUSA 变量 ------------------------------
#  只有 PLATFORM=moore 时才会用到, 取舍理由见下面 moore 分支的长注释.
#  MUSA_HOME   : MUSA SDK 安装根目录 (官方默认 /usr/local/musa)
#  MUSA_LIBDIR : 运行时库目录 (SDK 新老版本有 lib / lib64 两种布局)
#  MUSA_MODE   : 用哪条路消化"源码写 cudaXxx, 运行时叫 musaXxx"的差异.
#                auto(默认) / wrapper / mapping / plain, 见分支注释
#  MUSA_ARCH   : 目标架构代号; 留空交给 mcc 自己决定(MUSA 3.1.0 默认出
#                mp_21 + mp_22 两个). 不知道该填什么时先跑
#                scripts/run_domestic.sh moore, 它会自动试出可用的代号
MUSA_HOME      ?= /usr/local/musa
MUSA_LIBDIR    ?= $(MUSA_HOME)/lib
MUSAMAPPING    ?= $(MUSA_HOME)/tools/musaMapping
MUSA_MODE      ?= auto
MUSA_ARCH      ?=

# ------------------------------- 平台配置 ------------------------------------
#  CXX         : 设备编译器
#  ARCH_FLAGS  : 架构标志 (只有 nvcc 需要)
#  DEV_FLAGS   : 交给设备前端、不进主机编译器的参数
#  HOST_FLAGS  : 需要传给主机编译器的参数
#  LDFLAGS     : 链接参数
ifeq ($(PLATFORM),nvidia)
    CXX         := nvcc
    ARCH        ?= native
    ARCH_FLAGS  := -arch=$(ARCH)
    # -lineinfo 只附加调试行号表, 不影响代码生成, 但让 ncu/nsys 能定位到源码行
    DEV_FLAGS   := -lineinfo
    HOST_FLAGS  :=
    LDFLAGS     :=
else ifeq ($(PLATFORM),iluvatar)
    # 天数智芯 CoreX: 用 corex 自带的 clang++, 它按扩展名识别 .cu, 无需 -x cuda
    CXX         := clang++
    ARCH_FLAGS  :=
    DEV_FLAGS   :=
    HOST_FLAGS  := -I/usr/local/corex/include -fPIC
    LDFLAGS     := -L/usr/local/corex/lib64 -lcudart
else ifeq ($(PLATFORM),moore)
    # 摩尔线程 MUSA: mcc, SDK 默认安装在 /usr/local/musa
    #
    # 这是四个平台里与英伟达差得最远的一个, 而且差异不在源码语义上: MUSA 的
    # 运行时对象叫 musaXxx (musaMalloc / musaMemcpy / musa_runtime.h), 而本项目
    # 源码统一写 cudaXxx, 这样的调用点有 200 多处. 逐行改名既费时又容易改错,
    # 也没有必要 —— 官方对这件事给了三条现成的路, 都不需要动源码:
    #
    #   wrapper (默认首选)  mcc -mtgpu -cuda_wrapper ... -lcuda2musa
    #       mcc 的"兼容封装模式", 手册原话是"支持开发人员使用 CUDA API 而不是
    #       MUSA 的". MUSA SDK 3.1.0 与 5.2.0 的手册都写了这条, 也是我们在没有
    #       MUSA Mapping 的 MUSA 3.1.0 镜像上唯一可用的自动方案.
    #   mapping             mcc -x musa -fplugin=<libMusaMapping.so>
    #       SDK >= 5.2.0 才有的编译期 Clang 插件, 按映射表重写头文件名与标识符.
    #   plain               mcc -mtgpu ... -lmusart
    #       源码已经用官方离线工具迁移过才用这条:
    #           musify-text --inplace -- src/*.cu tests/*.cu   # 需 python3 + ahocorapy
    #
    # auto 的取舍顺序: 有 libcuda2musa.so 走 wrapper; 否则有 Mapping 插件走
    # mapping; 都没有就按 plain 编(那时源文件必须已经迁移过).
    #
    # 三条来自官方手册的注意事项:
    #   * .cu 后缀 mcc 默认不认, 必须 -mtgpu(或 -x musa) 才按 MUSA 源文件处理;
    #   * wrapper 模式要链接 libcuda2musa.so 而**不是** libmusart.so; 若报
    #     "未定义引用", 先试在 LDFLAGS 里补 -lmusart, 再检查 -l 的先后顺序;
    #   * mcc 默认按 --as-needed 链接, 顺序敏感; 必要时用 -Wl,--no-as-needed 覆盖.
    CXX         := mcc
    ARCH_FLAGS  :=
    ifeq ($(MUSA_MODE),auto)
        ifneq ($(wildcard $(MUSA_LIBDIR)/libcuda2musa.so),)
            MUSA_MODE := wrapper
        else ifneq ($(wildcard $(MUSAMAPPING)/libMusaMapping.so),)
            MUSA_MODE := mapping
        else
            MUSA_MODE := plain
        endif
    endif
    ifeq ($(MUSA_MODE),wrapper)
        DEV_FLAGS   := -mtgpu -cuda_wrapper
        LDFLAGS     := -L$(MUSA_LIBDIR) -lcuda2musa
    else ifeq ($(MUSA_MODE),mapping)
        DEV_FLAGS   := -x musa -I. -fplugin=$(MUSAMAPPING)/libMusaMapping.so
        export MUSAMAPPING_PATH  = $(MUSAMAPPING)
        export MUSA_INCLUDE_PATH = $(MUSA_HOME)/include
        LDFLAGS     := -L$(MUSA_LIBDIR) -Wl,--no-as-needed -lmusart -lmusa
    else
        DEV_FLAGS   := -mtgpu
        LDFLAGS     := -L$(MUSA_LIBDIR) -lmusart
    endif
    ifneq ($(MUSA_ARCH),)
        DEV_FLAGS   += --offload-arch=$(MUSA_ARCH)
    endif
    HOST_FLAGS  := -I$(MUSA_HOME)/include
    $(info [moore] MUSA_MODE=$(MUSA_MODE), MUSA_ARCH='$(MUSA_ARCH)')

else ifeq ($(PLATFORM),metax)
    # 沐曦 MACA: mxcc
    CXX         := mxcc
    ARCH_FLAGS  :=
    DEV_FLAGS   :=
    HOST_FLAGS  :=
    LDFLAGS     := -lmxcudart
else
    $(error 不支持的 PLATFORM '$(PLATFORM)' (可选: nvidia / iluvatar / moore / metax))
endif

# Windows 上的主编译器是 cl.exe: 必须显式声明源码为 UTF-8, 否则源码中的中文注释与
# 字符串会被按本地代码页(GBK)解释; /wd4129 屏蔽"字符字面量含非 ASCII 字符"的告警.
#
# 但 /utf-8 只作用于主机编译器 cl.exe, 管不到 nvcc 自己的 C++ 前端 cudafe++: 后者在
# 中文 Windows 上按系统 ANSI 代码页(CP936)读源码, 于是 UTF-8 的汉字会被当成 GBK
# 双字节错位切分, 偶发地把字符串的收尾引号吞成 GBK 尾字节, 报
#     error: missing closing quote
# 这个报错与汉字内容无关, 只与本行之前出现过多少字节有关 —— 在字符串里多打一个空格
# 就能"修好", 极易误诊成"nvcc 不支持中文". 根治办法是让前端也按 UTF-8 解码:
#     -Xcudafe --unicode_source_kind=UTF-8
# 但它不能单独使用(前端随后会改用系统代码页输出预处理结果, 反而不认自己的输入),
# 必须与上面的 /utf-8 同时给出. 两条合起来, "UTF-8 无 BOM 源码 + 中文字符串字面量"
# 才能在 Windows + nvcc 上稳定编译. Linux 前端(clang++ / mcc / mxcc)不需要这两条.
ifeq ($(OS),Windows_NT)
    HOST_FLAGS  += -Xcompiler /utf-8 -Xcompiler /wd4129 \
                   -Xcudafe --unicode_source_kind=UTF-8
endif

# ARCH_FLAGS 只有 nvidia 分支非空; DEV_FLAGS 在 nvidia 与 moore 两个分支非空.
# 两者都拼在源码之前, 因此 -x musa 之类作用于输入语言的开关能正常生效.
CXXFLAGS := -std=c++17 -O3 -Iinclude -Itests $(ARCH_FLAGS) $(DEV_FLAGS) $(HOST_FLAGS) \
            $(LPQ_EXTRA)

# 共同源文件 (不含各自的 main)
COMMON_SRC := src/config.cu src/lowp_tensor.cu src/quantize.cu \
              src/dequantize.cu src/metrics.cu src/reference.cu

APP_SRC  := src/main.cu $(COMMON_SRC)

APP_BIN   := $(BIN)/lpq_quant
SUITE_BIN := $(BIN)/test_formats $(BIN)/test_pipeline
PROBE_BIN := $(BIN)/test_portability
TEST_BIN  := $(SUITE_BIN) $(PROBE_BIN)
# 带宽天花板对照 kernel: 只依赖 CUDA 运行时, 四个平台都能编. 它给出该机器的
# 纯搬运上限, 用来把"实测带宽"换算成"相对这台机器可达上限的达成率" —— 换卡
# 之后重跑一次即可, 否则跨机器的 GB/s 数字没有可比性.
BW_BIN    := $(BIN)/bw_ceiling

.PHONY: all build app test probe run exp bw clean help

all: build test

build: app $(TEST_BIN)

app: $(APP_BIN)

# 探针放在最前面: 换平台后先确认平台能力齐全, 再去跑完整的正确性测试
test: $(TEST_BIN)
	@echo "=== test_portability (平台能力) ==="
	@$(PROBE_BIN)
	@echo "=== test_formats (格式与舍入) ==="
	@$(BIN)/test_formats
	@echo "=== test_pipeline (端到端管线) ==="
	@$(BIN)/test_pipeline

probe: $(PROBE_BIN)
	@$(PROBE_BIN)

$(APP_BIN): $(APP_SRC) $(wildcard include/lpq/*.hpp) $(wildcard include/lpq/*.cuh)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(APP_SRC) -o $@ $(LDFLAGS)

$(BIN)/test_formats: tests/test_formats.cu $(COMMON_SRC)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $< $(COMMON_SRC) -o $@ $(LDFLAGS)

$(BIN)/test_pipeline: tests/test_pipeline.cu $(COMMON_SRC)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $< $(COMMON_SRC) -o $@ $(LDFLAGS)

# 探针只依赖 common.hpp / platform.hpp / vec_io.cuh, 刻意不链接项目其余部分,
# 这样即使别处编译不过, 它仍能给出平台能力报告
$(PROBE_BIN): tests/test_portability.cu
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# 带宽天花板对照 kernel. 刻意不挂进 build 目标: 它是标定工具, 编不过也不该
# 挡住正确性测试与端到端实验
bw: $(BW_BIN)

$(BW_BIN): tools/bw_ceiling.cu
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# 快速跑一次示例: 生成 512x512 正态矩阵, 用 NVFP4 block 量化并逐位对拍 CPU 参考实现
run: app
	@mkdir -p data/tensors data/out
	python3 tools/gen_data.py --out-dir data/tensors --name normal_512 \
	        --rows 512 --cols 512 --dist normal --dtype fp32
	$(BIN)/lpq_quant --input data/tensors/normal_512.tensor \
	        --params data/params/nvfp4_block_fp16_nearest.params \
	        --out-dir data/out --cpu-check

# 复现全部实验: 260 次配置运行 + 规模扫描 + 汇总表格
exp: app
	python3 scripts/run_experiments.py
	python3 scripts/benchmark.py
	python3 scripts/cpu_compare.py
	python3 scripts/analyze.py > docs/measurements.md

clean:
	rm -rf $(BUILD_DIR)/*.o $(APP_BIN) $(TEST_BIN) $(BW_BIN)

help:
	@sed -n '2,18p' Makefile
