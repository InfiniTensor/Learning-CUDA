# Collect Nsight Systems profiles for the report.
#
# The GPU tracer path must not contain non-ASCII characters (the profiler's
# protobuf export fails on them), so every profile runs out of $env:TEMP\fhwt_prof
# and only the resulting .nsys-rep / .txt files are copied back into docs\profile.
$ErrorActionPreference = "Stop"
$Nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe"
$Root = Split-Path -Parent $PSScriptRoot
$Work = Join-Path $env:TEMP "fhwt_prof"
$Out = Join-Path $Root "docs\profile"

New-Item -ItemType Directory -Force -Path $Out | Out-Null
New-Item -ItemType Directory -Force -Path $Work | Out-Null
Copy-Item (Join-Path $Root "build\fhwt.exe") (Join-Path $Work "fhwt.exe") -Force
Set-Location $Work

# name -> arguments passed to fhwt.exe
$cases = [ordered]@{
    "dim256_dram"    = @("--mode","bench","--kernel","reg","--dim","256","--rows","262144","--dtype","f16","--iters","50","--warmup","10","--reps","1")
    "dim256_l2"      = @("--mode","bench","--kernel","reg","--dim","256","--rows","32768","--dtype","f16","--iters","50","--warmup","10","--reps","1")
    "dim1024_dram"   = @("--mode","bench","--kernel","reg","--dim","1024","--rows","65536","--dtype","f16","--iters","50","--warmup","10","--reps","1")
    "dim256_tc"      = @("--mode","bench","--kernel","tc","--dim","256","--rows","262144","--dtype","f16","--iters","50","--warmup","10","--reps","1")
    "quant_fp8_256"  = @("--mode","quant","--quant","fp8","--dim","256","--rows","262144","--dtype","f16","--iters","50","--warmup","10")
    "dim256_inplace" = @("--mode","bench","--kernel","reg","--dim","256","--rows","65536","--dtype","f16","--inplace","--iters","50","--warmup","10","--reps","1")
}

foreach ($name in $cases.Keys) {
    Write-Host "=== profiling $name ===" -ForegroundColor Cyan
    & $Nsys profile -t cuda --stats=false --force-overwrite=true -o $name `
        (Join-Path $Work "fhwt.exe") @($cases[$name]) | Out-Null
    $txt = & $Nsys stats --force-export=true `
        --report cuda_gpu_kern_sum,cuda_gpu_trace,cuda_kern_exec_sum,cuda_api_sum `
        "$name.nsys-rep" 2>&1
    [IO.File]::WriteAllText((Join-Path $Out "$name.txt"), ($txt -join "`n") + "`n",
                            (New-Object Text.UTF8Encoding($false)))
    Copy-Item "$name.nsys-rep" (Join-Path $Out "$name.nsys-rep") -Force
}
# --- launch-bound phase ------------------------------------------------------
# Small shapes: the kernel is only ~2.6 us, so the timeline is dominated by the
# host-side submission path rather than by the kernel itself.  Two burst lengths
# are profiled to show that the per-iteration cost grows with the burst length,
# and `tools/gap_analysis.py` turns the sqlite into docs\profile\launch_bound_analysis.md
# (report 8.5).
$small = [ordered]@{
    "small_iters20"   = @("--mode","bench","--kernel","reg","--dim","128","--rows","4096","--dtype","f16","--iters","20","--warmup","50","--reps","1")
    "small_iters2000" = @("--mode","bench","--kernel","reg","--dim","128","--rows","4096","--dtype","f16","--iters","2000","--warmup","50","--reps","1")
}
foreach ($name in $small.Keys) {
    Write-Host "=== profiling $name ===" -ForegroundColor Cyan
    & $Nsys profile -t cuda --stats=false --force-overwrite=true -o $name `
        (Join-Path $Work "fhwt.exe") @($small[$name]) | Out-Null
    # `--force-export` writes <name>.sqlite next to the .nsys-rep; the launch-bound
    # analysis reads that sqlite directly (kernel start/end plus the API call that
    # issued it), which the `nsys stats` text reports cannot express.
    & $Nsys stats --force-export=true --report cuda_gpu_kern_sum "$name.nsys-rep" | Out-Null
    Copy-Item "$name.nsys-rep" (Join-Path $Out "$name.nsys-rep") -Force
}
python (Join-Path $Root "tools\gap_analysis.py") `
    --sqlite (Join-Path $Work "small_iters20.sqlite") `
    --sqlite (Join-Path $Work "small_iters2000.sqlite") `
    --out (Join-Path $Out "launch_bound_analysis.md")

Write-Host "profiles written to $Out" -ForegroundColor Green
