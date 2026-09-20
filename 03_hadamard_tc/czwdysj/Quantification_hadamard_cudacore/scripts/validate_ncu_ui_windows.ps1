param([Parameter(Mandatory=$true)][string]$ReportDirectory)
$ErrorActionPreference = 'Stop'
$installs = Get-ChildItem 'C:\Program Files\NVIDIA Corporation' -Directory -Filter 'Nsight Compute *' | Sort-Object Name -Descending
if (-not $installs) { throw 'Windows Nsight Compute installation not found' }
$install = $installs[0].FullName
$ncu = Join-Path $install 'target\windows-desktop-win7-x64\ncu.exe'
$ui = Join-Path $install 'host\windows-desktop-win7-x64\ncu-ui.exe'
$log = Join-Path $ReportDirectory 'validation.log'
$reports = @('v0','v1','v2') | ForEach-Object { Join-Path $ReportDirectory "fused_${_}_fp16_b1024_d4096.ncu-rep" }

foreach ($version in @('v0','v1','v2')) {
    $report = Join-Path $ReportDirectory "fused_${version}_fp16_b1024_d4096.ncu-rep"
    $details = Join-Path $ReportDirectory "fused_${version}_windows_details.txt"
    "Windows import: $report" | Add-Content -Encoding utf8 $log
    & $ncu --import $report --page details --print-details header 2>&1 | Set-Content -Encoding utf8 $details
    if ($LASTEXITCODE -ne 0) { throw "Windows NCU import failed: $version" }
    $content = Get-Content $details -Raw
    if ($content -notmatch "fused_${version}_kernel" -or $content -notmatch 'Block Size\s+256' -or $content -notmatch 'Grid Size\s+1,?024') {
        throw "Windows report content validation failed: $version"
    }
    "Windows import PASS: $version, grid=1024, block=256" | Add-Content -Encoding utf8 $log
}

$args = ($reports | ForEach-Object { '"' + $_ + '"' }) -join ' '
$started = Start-Process -FilePath $ui -ArgumentList $args -PassThru
Start-Sleep -Seconds 10
$processes = Get-Process -Name ncu-ui -ErrorAction SilentlyContinue
if (-not $processes) { throw 'NCU-UI exited before loading the reports' }
"Windows NCU-UI PASS: reports=3, pid=$($processes.Id -join ','), title=$($processes.MainWindowTitle -join ' | ')" | Add-Content -Encoding utf8 $log

$manifestPath = Join-Path $ReportDirectory 'manifest.json'
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
$manifest | Add-Member -Force NoteProperty windows_ncu ((& $ncu --version 2>&1) -join "`n")
$manifest | Add-Member -Force NoteProperty windows_import_validated $true
$manifest | Add-Member -Force NoteProperty windows_ncu_ui_validated $true
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $manifestPath
