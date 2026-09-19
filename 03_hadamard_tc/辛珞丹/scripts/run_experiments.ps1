# =============================================================================
#  run_experiments.ps1 -- reproduce every table in the report on Windows.
#
#  Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1
#
#  Writes CSV / LOG artefacts into data\ (consumed by tools/analyze.py).
#
#  All timings are best-of-`Reps` bursts of `Iters` launches.  This box shares
#  its GPU with the desktop, so a single burst can be perturbed by unrelated GPU
#  work and read up to 2x low; the repetitions make the tables reproducible.
# =============================================================================
param(
    [string]$Bin = "build\fhwt.exe",
    [int]$Iters = 50,
    [int]$Warmup = 10,
    [int]$Reps = 3
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
# [IO.File] resolves relative paths against the *process* working directory,
# which is not changed by Set-Location.  Keep the two in sync so every artefact
# lands in <root>\data no matter where the script was launched from.
[Environment]::CurrentDirectory = $Root
New-Item -ItemType Directory -Force -Path data | Out-Null

if (-not (Test-Path $Bin)) { throw "binary not found: $Bin (run scripts\build.ps1 first)" }

# Windows PowerShell's `>` redirect and Tee-Object default to UTF-16, which the
# Python report generator cannot read.  Always go through this helper so every
# artefact on disk is BOM-less UTF-8.
function Write-Utf8 {
    param([string]$Path, [string[]]$Lines)
    [IO.File]::WriteAllText($Path, (($Lines -join "`n") + "`n"),
                            (New-Object Text.UTF8Encoding($false)))
}

function Invoke-Step {
    param([string]$Name, [string[]]$StepArgs, [string]$Log)
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $out = @(& $Bin @StepArgs 2>&1 | ForEach-Object { "$_" })
    Write-Utf8 $Log $out
    $out | Select-Object -Last 3 | Write-Host
    $sw.Stop()
    Write-Host ("    done ({0:n1}s)" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
}

# --- 0. environment ---------------------------------------------------------
$info = @(& $Bin --mode info | ForEach-Object { "$_" })
Write-Utf8 "data\device_info.txt" $info
$info | Write-Host

# --- 1. tuning sweep: launch geometry per (dim, rows) ------------------------
Invoke-Step "tune" @("--mode","tune","--rows","65536","--dtype","f16",
                     "--dims","64,128,256,512,1024",
                     "--iters","$Iters","--warmup","$Warmup","--reps","$Reps",
                     "--csv","data\tune.csv") "data\tune.log"

# --- 2. register / shuffle kernel matrix -------------------------------------
Invoke-Step "matrix/reg" @("--mode","matrix","--kernel","reg","--iters","$Iters",
                           "--warmup","$Warmup","--reps","$Reps",
                           "--csv","data\kernels_reg.csv") "data\kernels_reg.log"

# --- 3. tensor-core kernel matrix --------------------------------------------
Invoke-Step "matrix/tc" @("--mode","matrix","--kernel","tc","--iters","$Iters",
                          "--warmup","$Warmup","--reps","$Reps",
                          "--csv","data\kernels_tc.csv") "data\kernels_tc.log"

# --- 4. shared-memory kernel: large head dimensions --------------------------
Invoke-Step "matrix/smem" @("--mode","matrix","--kernel","smem",
                            "--dims","64,128,256,512,1024,2048,4096,8192,16384",
                            "--batch","2048,8192,32768,131072",
                            "--iters","$Iters","--warmup","$Warmup","--reps","$Reps",
                            "--csv","data\kernels_smem.csv") "data\kernels_smem.log"

# --- 5. in-place vs out-of-place: the L2 capacity effect ---------------------
Invoke-Step "inplace" @("--mode","inplace","--dims","128,256,512,1024",
                        "--dtypes","f16,bf16",
                        "--batch","8192,16384,32768,65536,131072,262144",
                        "--iters","100","--warmup","10","--reps","5",
                        "--csv","data\inplace.csv") "data\inplace.log"

# --- 6. fused quantisation ---------------------------------------------------
foreach ($q in @("fp8","int4")) {
    foreach ($d in @(64,128,256,512,1024)) {
        Write-Host "=== quant $q dim=$d ===" -ForegroundColor Cyan
        $out = @(& $Bin --mode quant --quant $q --dim $d --rows 262144 --dtype f16 `
                        --iters $Iters --warmup $Warmup 2>&1 | ForEach-Object { "$_" })
        Write-Utf8 "data\quant_${q}_$d.log" $out
        $out | Select-Object -Last 4 | Write-Host
    }
}

# --- 7. CUDA Graph vs raw launches ------------------------------------------
# The shape list and the equal-duration-burst logic live in tools/bench_graph.py
# so that the Linux (make) and Windows paths produce identical numbers.
python (Join-Path $Root "tools\bench_graph.py") --bin $Bin --out data

# --- 7b. how many launches are inside the graph (A / B / C / D) --------------
# A and B measure the host submission rate, which the desktop load moves by more
# than 3x between sweeps; five sweeps are taken and the median is reported (only
# the C column is a stable reading).
python (Join-Path $Root "tools\bench_loop.py") --bin $Bin --out data --reps 3 --runs 5

# --- 7b'. how many *roots* the recorded graph has (report 5.8) ---------------
# The two variants are interleaved (all-roots, single-root, ...) so a drift in the
# machine lands on both columns instead of only one.
python (Join-Path $Root "tools\bench_roots.py") --bin $Bin --out data --reps 3 --runs 5

# --- 7c. rotation fused into the GEMM's A-tile load (report 7.12) ------------
python (Join-Path $Root "tools\bench_gemm.py") --bin $Bin --out data --reps 3 --runs 5
# Fork/join *width* inside a layer: N independent tensors per layer, plus the
# multi-stream control (report 7.13).
python (Join-Path $Root "tools\bench_group.py") --bin $Bin --out data --reps 3 --runs 3
# How many passes one graph should hold: K = 1, 2, 5, 10, 20 passes per graph
# (report 7.14).  K must divide --iters, otherwise the window's tail is dropped.
python (Join-Path $Root "tools\bench_batch.py") --bin $Bin --out data --reps 3 --runs 8


# --- 8. CPU baseline ---------------------------------------------------------
Invoke-Step "cpu" @("--mode","cpu","--dim","256","--rows","65536","--threads","8") "data\cpu.log"

# --- 8b. call-site advice (in-place / batching heuristics) -------------------
$advice = @()
foreach ($sh in @(@(1024,64), @(4096,128), @(16384,256), @(65536,256), @(131072,256), @(262144,512))) {
    $advice += @(& $Bin --mode advice --rows $sh[0] --dim $sh[1] --dtype f16 2>&1 | ForEach-Object { "$_" })
    $advice += ""
}
Write-Utf8 "data\advice.log" $advice
Write-Host "=== advice ===" -ForegroundColor Cyan
$advice | Select-Object -First 8 | Write-Host

# --- 9. correctness suite ----------------------------------------------------
$lines = @()
foreach ($t in @("test_correctness","test_fusion","test_fp8","test_graph","test_gemm")) {
    $exe = Join-Path (Split-Path -Parent $Bin) "$t.exe"
    $out = @(& $exe 2>&1 | ForEach-Object { "$_" })
    $lines += "$t : " + ($out | Select-Object -Last 1)
    $lines[-1] | Write-Host
}
Write-Utf8 "data\tests.log" $lines

Write-Host "`nall experiments finished" -ForegroundColor Green
