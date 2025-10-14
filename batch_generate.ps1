<#
.SYNOPSIS
    批量调用 generate_elliptic_composites.py 脚本来生成多个不同的复合材料结构样本。
.DESCRIPTION
    此脚本自动化了随机结构生成过程，支持并行执行以提高效率。
    它会为每个样本使用一个唯一的随机种子，并将所有输出文件（JSON 和一个汇总的 CSV）
    整理到指定的输出目录中。
.NOTES
    作者: [Your Name]
    日期: [Current Date]
    版本: 1.1

    前置要求:
    1. PowerShell 5.1 或更高版本 (Windows 10/11 自带)。
    2. Python 已经安装并配置在系统 PATH 中。
    3. generate_elliptic_composites.py 脚本与本脚本位于同一目录下或在指定路径。
#>

# --- 参数配置区 ---

# Python 脚本的路径
$pythonScriptPath = ".\generate_elliptic_composites.py"

# 输出文件的根目录
$outputBaseDir = ".\generated_samples"

# 目标生成的样本总数
$numberOfSamples = 200

# 并行执行的任务数 (建议设置为 CPU 核心数或稍小)
# 可以通过 (Get-WmiObject -class Win32_Processor).NumberOfLogicalProcessors 查看逻辑核心数
$maxParallelJobs = 8

# --- Python 脚本参数配置 ---
# 这里定义的参数会传递给每个 Python 进程。
# 对于需要变化的参数（如 --seed），我们会在循环中动态生成。

# 几何与材料参数
$Lx = 1.0
$Ly = 1.0
$km = 1.0
$ki = 10.0

# 椭圆生成控制 (二选一或共同作用)
# 注意：如果像原脚本那样，phi_target 会在 N 之后继续添加，直到满足条件。
# 如果只想按 N 生成，将 $phiTarget 设为 $null
$N = 40
$phiTarget = 0.35

# 椭圆几何约束
$gmin = 0.002
$boundary_margin = 0.01

# 椭圆尺寸与角度范围
$a_min = 0.02
$a_max = 0.08
$b_min = 0.01
$b_max = 0.04
$theta_min = 0.0
$theta_max = 180.0

# --- 脚本执行区 ---

# 检查 Python 脚本是否存在
if (-not (Test-Path $pythonScriptPath)) {
    Write-Error "错误: Python 脚本未找到于 '$pythonScriptPath'。请检查路径。"
    exit 1
}

# 准备输出目录
$jsonOutputDir = Join-Path -Path $outputBaseDir -ChildPath "json_files"
$summaryCsvPath = Join-Path -Path $outputBaseDir -ChildPath "samples_summary.csv"

# 创建目录 (如果不存在)
New-Item -ItemType Directory -Force -Path $jsonOutputDir | Out-Null
Write-Host "输出目录已准备好: $jsonOutputDir"

# 如果汇总 CSV 文件已存在，则删除，以便从头开始创建一个干净的文件
if (Test-Path $summaryCsvPath) {
    Remove-Item $summaryCsvPath
    Write-Host "已删除旧的汇总 CSV 文件: $summaryCsvPath"
}

# 开始计时
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host "开始生成 $numberOfSamples 个样本，并行度为 $maxParallelJobs ..."
Write-Host "--------------------------------------------------------"

# 循环生成所有样本
for ($i = 1; $i -le $numberOfSamples; $i++) {
    
    # 使用一个随机数作为种子，确保每次运行脚本生成的样本集都不同
    # 如果需要可复现的批量，可以将种子设置为 $i
    $seed = Get-Random -Minimum 1000 -Maximum 999999

    # 构建 Python 脚本的参数列表
    $arguments = @(
        "--out_dir", $jsonOutputDir,
        "--summary_csv", $summaryCsvPath,
        "--seed", $seed,
        "--Lx", $Lx,
        "--Ly", '1.0', # Ly.ToString()
        "--km", $km,
        "--ki", $ki,
        "--N", $N,
        "--phi_target", $phiTarget,
        "--gmin", $gmin,
        "--boundary_margin", $boundary_margin,
        "--a_min", $a_min,
        "--a_max", $a_max,
        "--b_min", $b_min,
        "--b_max", $b_max,
        "--theta_min", $theta_min,
        "--theta_max", $theta_max,
        "--id_prefix", "sample_setA" # 可以为不同批次设置不同前缀
    )

    # PowerShell 的后台作业机制
    # 当正在运行的作业数量达到上限时，等待任一作业完成
    while ((Get-Job -State Running).Count -ge $maxParallelJobs) {
        # 等待 1 秒再检查，避免 CPU 空转
        Start-Sleep -Seconds 1 
    }

    # 开始一个新的后台作业来运行 Python 脚本
    # -ArgumentList 需要一个数组
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] 启动任务 $i / $numberOfSamples (Seed: $seed)"
    Start-Job -ScriptBlock {
        param($pyScript, $args)
        # 在作业中执行 python 命令
        python $pyScript @args
    } -ArgumentList @($pythonScriptPath, $arguments) | Out-Null # Out-Null 隐藏作业对象的输出
}

# 所有作业都已启动，现在等待它们全部完成
Write-Host "所有 $numberOfSamples 个任务均已启动，正在等待完成..."
Get-Job | Wait-Job | Out-Null

# 检查作业是否有错误并清理
Write-Host "所有任务已完成。正在收集结果..."
$failedJobs = 0
foreach ($job in (Get-Job)) {
    if ($job.State -eq 'Failed') {
        $failedJobs++
        Write-Warning "作业 $($job.Id) 失败。"
        # 输出该作业的错误信息
        Receive-Job -Job $job
    } else {
        # 成功作业的输出可以忽略，因为 Python 脚本自己会写入文件
        # Receive-Job -Job $job | Out-Null (如果需要查看python的print输出，可以取消注释)
    }
}

# 清理所有已完成的作业对象
Get-Job | Remove-Job

# 停止计时并报告结果
$stopwatch.Stop()
$elapsedSeconds = $stopwatch.Elapsed.TotalSeconds
$avgTimePerSample = $elapsedSeconds / $numberOfSamples

Write-Host "--------------------------------------------------------"
Write-Host "批量生成完成！"
Write-Host "总计生成样本数: $numberOfSamples"
Write-Host "成功 / 失败: $($numberOfSamples - $failedJobs) / $failedJobs"
Write-Host "JSON 文件保存于: $jsonOutputDir"
Write-Host "汇总数据文件:  $summaryCsvPath"
Write-Host "总耗时: $('{0:N2}' -f $elapsedSeconds) 秒"
Write-Host "平均每个样本耗时: $('{0:N2}' -f $avgTimePerSample) 秒"
Write-Host "--------------------------------------------------------"

