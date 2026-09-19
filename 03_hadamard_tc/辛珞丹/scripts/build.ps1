# =============================================================================
#  build.ps1 -- Windows development build (the Makefile is the reference build
#  for the final submission on Linux).
# =============================================================================
param(
    [string]$Arch = "native",
    [ValidateSet("all", "tests", "app")][string]$Target = "all",
    [switch]$Clean,
    [switch]$Fast
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Build = Join-Path $Root "build"
$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $VcVars)) { throw "vcvars64.bat not found: $VcVars" }
if ($Clean -and (Test-Path $Build)) { Remove-Item -Recurse -Force $Build }
New-Item -ItemType Directory -Force -Path $Build | Out-Null

$Common = @("src/ops.cu")
$Flags = @(
    "-std=c++17", "-O3", "-Iinclude", "-I.", "-arch=$Arch",
    "-Xcompiler", "/utf-8", "-Xcompiler", "/wd4129",
    "-Xcudafe", "--unicode_source_kind=UTF-8",
    "-diag-suppress", "192,177,20012",
    "-lineinfo"
)
if (-not $Fast) { $Flags += "-lineinfo" }

$jobs = @()
if ($Target -eq "all" -or $Target -eq "app") {
    $jobs += ,(@("src/main.cu") + $Common + @("-o", "build/fhwt.exe"))
}
if ($Target -eq "all" -or $Target -eq "tests") {
    $jobs += ,(@("tests/test_correctness.cu") + $Common + @("-o", "build/test_correctness.exe"))
    $jobs += ,(@("tests/test_fusion.cu") + $Common + @("-o", "build/test_fusion.exe"))
    $jobs += ,(@("tests/test_fp8.cu") + $Common + @("-o", "build/test_fp8.exe"))
    $jobs += ,(@("tests/test_graph.cu") + $Common + @("-o", "build/test_graph.exe"))
    $jobs += ,(@("tests/test_gemm.cu") + $Common + @("-o", "build/test_gemm.exe"))
}
foreach ($j in $jobs) {
    $cmdline = ($Flags + $j) -join " "
    Write-Host "=== nvcc $cmdline ===" -ForegroundColor Cyan
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & cmd.exe /c "call `"$VcVars`" >nul && cd /d `"$Root`" && nvcc $cmdline"
    $sw.Stop()
    if ($LASTEXITCODE -ne 0) { throw "build failed: $cmdline" }
    Write-Host ("    ok ({0:n1}s)" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
}
