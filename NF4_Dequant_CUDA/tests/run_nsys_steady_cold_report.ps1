param(
    [string]$WeightsBin = "tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin",
    [string]$ParamsTemplate = "tests/data/params_nsys_pinned.txt",
    [string]$OutputDir = "tests/data",
    [string]$ExePath = "build/Release/nf4_dequant.exe",
    [string]$NsysPath = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.2\target-windows-x64\nsys.exe",
    [int]$Rounds = 5,
    [int]$SteadyProfileLoopIters = 6
)

$ErrorActionPreference = "Stop"

Write-Host "[1/2] Running steady-state median report..."
& tests/run_nsys_reuse_compare.ps1 `
    -WeightsBin $WeightsBin `
    -ParamsTemplate $ParamsTemplate `
    -OutputDir $OutputDir `
    -ExePath $ExePath `
    -NsysPath $NsysPath `
    -Rounds $Rounds `
    -ProfileLoopIters $SteadyProfileLoopIters `
    -UseCudaProfilerRange:$true `
    -RunTag "steady"
if ($LASTEXITCODE -ne 0) {
    throw "steady-state report failed"
}

Write-Host "[2/2] Running cold-start median report..."
& tests/run_nsys_reuse_compare.ps1 `
    -WeightsBin $WeightsBin `
    -ParamsTemplate $ParamsTemplate `
    -OutputDir $OutputDir `
    -ExePath $ExePath `
    -NsysPath $NsysPath `
    -Rounds $Rounds `
    -ProfileLoopIters 1 `
    -UseCudaProfilerRange:$false `
    -RunTag "cold"
if ($LASTEXITCODE -ne 0) {
    throw "cold-start report failed"
}

$steadySummary = Join-Path $OutputDir "nsys_steady_median.csv"
$coldSummary = Join-Path $OutputDir "nsys_cold_median.csv"
Write-Host "Done."
Write-Host "Steady summary: $steadySummary"
Write-Host "Cold summary:   $coldSummary"
