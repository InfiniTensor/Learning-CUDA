param(
    [string]$WeightsBin = "tests/data/nf4_r4096_c4096_bs64_bpg256_weights.bin",
    [string]$ParamsTemplate = "tests/data/params_nsys_pinned.txt",
    [string]$OutputDir = "tests/data",
    [string]$ExePath = "build/Release/nf4_dequant.exe",
    [string]$NsysPath = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.2\target-windows-x64\nsys.exe",
    [int]$Rounds = 5,
    [int]$ProfileLoopIters = 6,
    [bool]$UseCudaProfilerRange = $true,
    [string]$RunTag = "steady"
)

$ErrorActionPreference = "Stop"

function Write-ParamsWithReuseFlag {
    param(
        [string]$InputPath,
        [string]$OutputPath,
        [bool]$ReuseDeviceBuffers,
        [int]$ProfileLoopIters
    )

    if (-not (Test-Path $InputPath)) {
        throw "params template not found: $InputPath"
    }

    $lines = Get-Content $InputPath
    $foundReuse = $false
    $foundKernelWarmup = $false
    $foundProfileLoop = $false
    $out = @()
    foreach ($line in $lines) {
        if ($line -match "^\s*reuse_device_buffers\s*=") {
            $out += "reuse_device_buffers = " + ($(if ($ReuseDeviceBuffers) { "true" } else { "false" }))
            $foundReuse = $true
        } elseif ($line -match "^\s*kernel_warmup_iters\s*=") {
            $out += "kernel_warmup_iters = 0"
            $foundKernelWarmup = $true
        } elseif ($line -match "^\s*profile_loop_iters\s*=") {
            $out += "profile_loop_iters = $ProfileLoopIters"
            $foundProfileLoop = $true
        } else {
            $out += $line
        }
    }
    if (-not $foundReuse) {
        $out += "reuse_device_buffers = " + ($(if ($ReuseDeviceBuffers) { "true" } else { "false" }))
    }
    if (-not $foundKernelWarmup) {
        $out += "kernel_warmup_iters = 0"
    }
    if (-not $foundProfileLoop) {
        $out += "profile_loop_iters = $ProfileLoopIters"
    }
    Set-Content -Path $OutputPath -Value $out
}

if (-not (Test-Path $WeightsBin)) {
    throw "weights bin not found: $WeightsBin"
}
if (-not (Test-Path $ExePath)) {
    throw "executable not found: $ExePath"
}
if (-not (Test-Path $NsysPath)) {
    throw "nsys.exe not found: $NsysPath"
}
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}
if ($Rounds -le 0) {
    throw "Rounds must be > 0"
}
if ($ProfileLoopIters -le 0) {
    throw "ProfileLoopIters must be > 0."
}
if ($UseCudaProfilerRange -and $ProfileLoopIters -le 1) {
    throw "When UseCudaProfilerRange=true, ProfileLoopIters must be > 1."
}

$baseStems = @()
$newStems = @()

if ($UseCudaProfilerRange) {
    Write-Host "[0/4][$RunTag] Running $Rounds rounds, profile_loop_iters=$ProfileLoopIters (capture last $($ProfileLoopIters - 1))."
} else {
    Write-Host "[0/4][$RunTag] Running $Rounds rounds, profile_loop_iters=$ProfileLoopIters (capture-range disabled)."
}
for ($r = 1; $r -le $Rounds; ++$r) {
    $paramsOff = Join-Path $OutputDir ("params_nsys_{0}_reuse_off_r{1}.txt" -f $RunTag, $r)
    $paramsOn = Join-Path $OutputDir ("params_nsys_{0}_reuse_on_r{1}.txt" -f $RunTag, $r)
    $reportOff = Join-Path $OutputDir ("nsys_{0}_reuse_off_r{1}" -f $RunTag, $r)
    $reportOn = Join-Path $OutputDir ("nsys_{0}_reuse_on_r{1}" -f $RunTag, $r)
    $outOff = Join-Path $OutputDir ("out_nsys_{0}_reuse_off_r{1}.bin" -f $RunTag, $r)
    $outOn = Join-Path $OutputDir ("out_nsys_{0}_reuse_on_r{1}.bin" -f $RunTag, $r)

    Write-Host "[1/4][$RunTag][Round $r/$Rounds] Generating params..."
    Write-ParamsWithReuseFlag -InputPath $ParamsTemplate -OutputPath $paramsOff -ReuseDeviceBuffers:$false -ProfileLoopIters $ProfileLoopIters
    Write-ParamsWithReuseFlag -InputPath $ParamsTemplate -OutputPath $paramsOn -ReuseDeviceBuffers:$true -ProfileLoopIters $ProfileLoopIters

    Write-Host "[2/4][$RunTag][Round $r/$Rounds] Nsight Systems profile (reuse=false)..."
    & tests/run_nsys_profile.ps1 `
        -WeightsBin $WeightsBin `
        -ParamsFile $paramsOff `
        -OutputBin $outOff `
        -ReportStem $reportOff `
        -ExePath $ExePath `
        -NsysPath $NsysPath `
        -WarmupFirst:$false `
        -UseCudaProfilerRange:$UseCudaProfilerRange
    if ($LASTEXITCODE -ne 0) {
        throw "nsys profile failed for reuse=false, round=$r"
    }
    if (-not (Test-Path "$reportOff.nsys-rep")) {
        throw "missing report: $reportOff.nsys-rep"
    }

    Write-Host "[3/4][$RunTag][Round $r/$Rounds] Nsight Systems profile (reuse=true)..."
    & tests/run_nsys_profile.ps1 `
        -WeightsBin $WeightsBin `
        -ParamsFile $paramsOn `
        -OutputBin $outOn `
        -ReportStem $reportOn `
        -ExePath $ExePath `
        -NsysPath $NsysPath `
        -WarmupFirst:$false `
        -UseCudaProfilerRange:$UseCudaProfilerRange
    if ($LASTEXITCODE -ne 0) {
        throw "nsys profile failed for reuse=true, round=$r"
    }
    if (-not (Test-Path "$reportOn.nsys-rep")) {
        throw "missing report: $reportOn.nsys-rep"
    }

    $baseStems += $reportOff
    $newStems += $reportOn
}

Write-Host "[4/4][$RunTag] Median comparison across $Rounds rounds..."
$compareOutput = & python tests/compare_nsys_csv.py `
    --base-stems ($baseStems -join ",") `
    --new-stems ($newStems -join ",")
if ($LASTEXITCODE -ne 0) {
    throw "compare_nsys_csv failed"
}
$summaryCsv = Join-Path $OutputDir ("nsys_{0}_median.csv" -f $RunTag)
$compareOutput | Set-Content -Path $summaryCsv

Write-Host "Done."
Write-Host "Median summary: $summaryCsv"
foreach ($line in $compareOutput) {
    Write-Host $line
}
Write-Host "Report stems (reuse=false):"
foreach ($s in $baseStems) {
    Write-Host "  $s"
}
Write-Host "Report stems (reuse=true):"
foreach ($s in $newStems) {
    Write-Host "  $s"
}
