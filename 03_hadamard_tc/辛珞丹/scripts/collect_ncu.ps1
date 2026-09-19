# =============================================================================
#  collect_ncu.ps1 -- Nsight Compute (ncu) 采集脚本
#
#  与 docs/profile/ 里的 nsys 采集采**同一批形状**，两个工具的数据可以互相印证：
#  nsys 给"什么时候发生了什么、内核之间有没有空隙"，ncu 给"内核内部为什么是这个速度"
#  （访存吞吐占峰值多少、发射了几个 warp、卡在什么 stall 上、占用率多少）。
#
#  用法（普通或管理员窗口都可以，前提是性能计数器权限已开）：
#      powershell -ExecutionPolicy Bypass -File scripts\collect_ncu.ps1
#      powershell -ExecutionPolicy Bypass -File scripts\collect_ncu.ps1 -Only dim256_dram
#
#  权限：ncu 需要访问 GPU 性能计数器。两种开法，二选一：
#    1) 管理员执行（默认 RmProfilingAdminOnly=1 时这就够了）：
#         右键 PowerShell -> 以管理员身份运行，再跑本脚本
#    2) 把计数器开放给所有用户（写完需要重启驱动/系统才生效）：
#         reg add "HKLM\SOFTWARE\NVIDIA Corporation\Global\NVTweak" `
#             /v RmProfilingAdminOnly /t REG_DWORD /d 0 /f
#       或 NVIDIA 控制面板 -> 桌面 -> 启用开发者设置 -> 管理 GPU 性能计数器
#  两种都拿不到时会得到 ERR_NVGPUCTRPERM，脚本会直接停下并打印上面这段话。
#
#  输出：docs/profile/ncu/<name>.txt（details 页）+ <name>.csv + _run.log
# =============================================================================
param(
    [string]$Bin = "",
    [string]$Out = "",
    [string]$Only = ""
)
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Bin) { $Bin = Join-Path $Root "build\fhwt.exe" }
if (-not $Out) { $Out = Join-Path $Root "docs\profile\ncu" }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Log = Join-Path $Out "_run.log"
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\target\windows-desktop-win7-x64\ncu.exe"
if (-not (Test-Path $ncu)) {
    $ncu = (Get-ChildItem "C:\Program Files\NVIDIA Corporation" -Directory -Filter "Nsight Compute*" |
            Sort-Object Name -Descending | Select-Object -First 1 |
            ForEach-Object { Join-Path $_.FullName "target\windows-desktop-win7-x64\ncu.exe" })
}
function Say($m) { $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m; Write-Host $line; Add-Content -Path $Log -Value $line }

Say "ncu      : $ncu"
Say "target   : $Bin"
Say "out dir  : $Out"
Say ("elevated : {0}" -f (New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))

# --- 0. 权限自检：先打一个最小的采集，确认计数器可用 -------------------------
Say "--- permission probe ---"
$probe = & $ncu --metrics gpu__time_duration.sum --launch-count 1 --target-processes all `
    $Bin --mode bench --dim 64 --rows 1024 --iters 1 --warmup 0 2>&1 | Out-String
$probe | Set-Content -Path (Join-Path $Out "_permission_probe.txt") -Encoding UTF8
if ($probe -match "ERR_NVGPUCTRPERM") {
    Say "PERMISSION DENIED (ERR_NVGPUCTRPERM) -- 见文件头部说明，两种开法二选一"
    Say "probe output saved to _permission_probe.txt"
    exit 1
}
if ($probe -match "==ERROR==") {
    Say "probe failed for another reason, see _permission_probe.txt"
    exit 1
}
Say "permission OK"

# --- 1. 采集清单：与 nsys 采过的形状一一对应 --------------------------------
#  name              : 输出文件名
#  args              : 传给 fhwt.exe 的参数
#  说明              : 这条想回答什么问题
$presets = @(
    # 形状/参数与 scripts/collect_profiles.ps1 里 nsys 那一批**逐字对齐**，
    # 这样同一个用例的 nsys 时间线与 ncu 计数器说的是同一个内核、同一个 shape。
    @{ name = "dim256_dram";    args = @("--mode","bench","--kernel","reg","--dim","256","--rows","262144","--dtype","f16","--iters","1","--warmup","1");
       why  = "DRAM 受限的主力形状：确认访存吞吐贴住实测 memcpy 屋顶" },
    @{ name = "dim256_l2";      args = @("--mode","bench","--kernel","reg","--dim","256","--rows","32768","--dtype","f16","--iters","1","--warmup","1");
       why  = "工作集驻留 L2：此时瓶颈应从 DRAM 转向 L1/指令，stall 分布应当不同" },
    @{ name = "dim1024_dram";   args = @("--mode","bench","--kernel","reg","--dim","1024","--rows","65536","--dtype","f16","--iters","1","--warmup","1");
       why  = "大 head_dim：每个 lane 持有的元素最多，寄存器压力最大的一档" },
    @{ name = "dim256_tc";      args = @("--mode","bench","--kernel","tc","--dim","256","--rows","262144","--dtype","f16","--iters","1","--warmup","1");
       why  = "Tensor Core 路径：验证它与寄存器路径的差距不在算力而在访存/重排" },
    @{ name = "dim256_inplace"; args = @("--mode","bench","--kernel","reg","--dim","256","--rows","65536","--dtype","f16","--inplace","--iters","1","--warmup","1");
       why  = "原地变换：访存量减半后，算力占比应当上升" },
    @{ name = "quant_fp8_256";  args = @("--mode","quant","--quant","fp8","--dim","256","--rows","262144","--dtype","f16","--iters","1","--warmup","1");
       why  = "融合量化：少一次内核、少一个中间张量，访存量 5 B/elem -> 3 B/elem" }
)

$sections = @("SpeedOfLight","MemoryWorkloadAnalysis","MemoryWorkloadAnalysis_Tables",
              "LaunchStats","Occupancy","SchedulerStats","WarpStateStats")

$ok = 0; $fail = 0
foreach ($p in $presets) {
    if ($Only -and $p.name -ne $Only) { continue }
    $txt = Join-Path $Out ($p.name + ".txt")
    $csv = Join-Path $Out ($p.name + ".csv")
    $secArgs = @(); foreach ($s in $sections) { $secArgs += @("--section", $s) }
    Say ("--- {0} :: {1}" -f $p.name, $p.why)
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $raw = & $ncu @secArgs --launch-count 1 --target-processes all --print-details all --kernel-name "regex:fhwt" `
           --csv $Bin @($p.args) 2>&1 | Out-String
    $sw.Stop()
    $raw | Set-Content -Path $txt -Encoding UTF8
    if ($raw -match "ERR_NVGPUCTRPERM") { Say ("  {0}: permission lost mid-run" -f $p.name); $fail++; continue }
    Say ("  {0}: {1:n1}s, output {2} bytes" -f $p.name, $sw.Elapsed.TotalSeconds, $raw.Length)
    $ok++
}
Say "done: ok=$ok fail=$fail, files in $Out"
