# =============================================================================
#  build.ps1 -- Windows 本地开发用构建脚本 (正式提交以 Makefile 为准)
#
#  用法:
#    powershell -ExecutionPolicy Bypass -File scripts\build.ps1 [-Arch native] [-Target all]
#
#  注意:
#    * 不使用 --use_fast_math, 因为本次实现的正确性判据是"与 CPU 参考实现逐位一致",
#      快速数学会破坏 IEEE754 语义 (重结合 / 近似除法);
#    * 源码必须是 UTF-8 无 BOM; 中文源码路径依赖主机编译器的 /utf-8;
#    * /wd4129 用于屏蔽 MSVC 对 UTF-8 字符字面量的告警;
#    * -lineinfo 只加调试行号表, 不影响代码生成, 但可让 ncu/nsys 定位到源码行.
# =============================================================================
param(
    [string]$Arch = "native",
    [string]$Target = "all",       # all | app | tests
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build"
$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path $VcVars)) { throw "找不到 vcvars64.bat: $VcVars" }
if ($Clean -and (Test-Path $BuildDir)) { Remove-Item -Recurse -Force $BuildDir }
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$Common = @(
    "src/config.cu", "src/lowp_tensor.cu", "src/quantize.cu",
    "src/dequantize.cu", "src/metrics.cu", "src/reference.cu"
)
$Flags = @(
    "-std=c++17", "-O3", "-Iinclude", "-arch=$Arch", "-lineinfo",
    "-Xcompiler", "/utf-8", "-Xcompiler", "/wd4129",
    # cudafe++ 默认按系统 ANSI 代码页(CP936)读源码, 会把 UTF-8 汉字按 GBK 错位
    # 切分并吞掉字符串收尾引号(报 missing closing quote); 必须显式声明 UTF-8.
    "-Xcudafe", "--unicode_source_kind=UTF-8",
    "-diag-suppress", "192,177"
)
$jobs = @()

if ($Target -eq "all" -or $Target -eq "app") {
    $jobs += ,(@("src/main.cu") + $Common + @("-o", "build/lpq_quant.exe"))
}
if ($Target -eq "all" -or $Target -eq "tests") {
    $jobs += ,(@("tests/test_formats.cu") + $Common + @("-o", "build/test_formats.exe"))
    $jobs += ,(@("tests/test_pipeline.cu") + $Common + @("-o", "build/test_pipeline.exe"))
    # 探针刻意不链接项目其余部分 (与 Makefile 的 probe 目标一致), 这样即使别处编不过,
    # 它仍能给出平台能力报告 -- 换平台后它是第一个要跑的东西.
    $jobs += ,(@("tests/test_portability.cu") + @("-o", "build/test_portability.exe"))
}

foreach ($j in $jobs) {
    $cmdline = ($Flags + $j) -join " "
    Write-Host "=== nvcc $cmdline ===" -ForegroundColor Cyan
    $cmd = "call `"$VcVars`" >nul && cd /d `"$Root`" && nvcc $cmdline"
    & cmd.exe /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "编译失败: $cmdline" }
}
Write-Host "构建完成 -> $BuildDir" -ForegroundColor Green
