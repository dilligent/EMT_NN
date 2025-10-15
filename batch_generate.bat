@ECHO OFF
TITLE Batch Structure Generator

:: 启用延迟变量展开，这对于在循环中使用变量非常重要
SETLOCAL EnableDelayedExpansion

:: ==========================================================================
:: 批处理脚本，用于批量调用 generate_elliptic_composites.py
:: ==========================================================================
:: 描述:
::   此脚本通过循环多次调用Python脚本，为每次调用生成一个唯一的随机种子，
::   从而创建大量不同的几何结构样本。
::
:: 修复说明:
::   - 启用延迟变量展开以确保RANDOM在每次循环中都重新生成
::   - 修复了变量作用域问题
::
:: 使用方法:
::   1. 将此 .bat 文件与 generate_elliptic_composites.py 放在同一个文件夹中。
::   2. 修改下面的“参数配置区”。
::   3. 双击运行此 .bat 文件。
:: ==========================================================================


:: --- 参数配置区 ---

:: 设置Python脚本的文件名
SET "PYTHON_SCRIPT=generate_elliptic_composites.py"

:: 设置输出文件的主目录
SET "OUTPUT_BASE_DIR=generated_samples_bat"

:: 设置要生成的样本总数
SET "NUM_SAMPLES=200"

:: --- Python脚本参数配置 ---
SET "Lx=1.0"
SET "Ly=1.0"
SET "km=1.0"
SET "ki=10.0"

SET "N=40"
SET "PHI_TARGET=0.35"

SET "GMIN=0.002"
SET "BOUNDARY_MARGIN=0.01"

SET "A_MIN=0.02"
SET "A_MAX=0.08"
SET "B_MIN=0.01"
SET "B_MAX=0.04"
SET "THETA_MIN=0.0"
SET "THETA_MAX=180.0"


:: --- 脚本执行区 (一般无需修改) ---

ECHO.
ECHO ===================================================
ECHO      批量样本生成器 (CMD Batch Version - Fixed)
ECHO ===================================================
ECHO.

:: 检查Python脚本是否存在
IF NOT EXIST "!PYTHON_SCRIPT!" (
    ECHO 错误: Python脚本 "!PYTHON_SCRIPT!" 不存在!
    ECHO 请确保此批处理文件和Python脚本在同一个目录下。
    PAUSE
    EXIT /B 1
)

:: 准备输出目录
SET "JSON_OUTPUT_DIR=!OUTPUT_BASE_DIR!\json_files"
SET "SUMMARY_CSV_PATH=!OUTPUT_BASE_DIR!\samples_summary.csv"

ECHO 准备输出目录: !JSON_OUTPUT_DIR!
IF NOT EXIST "!JSON_OUTPUT_DIR!" (
    MKDIR "!JSON_OUTPUT_DIR!"
)

:: 删除旧的汇总文件，确保每次运行都是全新的开始
IF EXIST "!SUMMARY_CSV_PATH!" (
    DEL "!SUMMARY_CSV_PATH!"
    ECHO 已删除旧的汇总CSV文件。
)

ECHO.
ECHO 即将开始生成 %NUM_SAMPLES% 个样本...
ECHO.
timeout /t 3 >nul

:: 主循环
:: FOR /L %%i IN (start, step, end) DO (command)
FOR /L %%i IN (1, 1, %NUM_SAMPLES%) DO (
    
    :: 使用延迟展开语法 !RANDOM! 来确保每次循环都生成新的随机数
    SET "CURRENT_SEED=!RANDOM!"
    
    ECHO 正在处理样本 %%i / %NUM_SAMPLES% ^(使用种子: !CURRENT_SEED!^)
    
    :: 调用Python脚本并传递所有参数
    :: 注意: CMD中没有--verbose，但我们可以通过不重定向输出达到类似效果
    :: 使用延迟展开语法 !变量! 来获取所有变量的当前值
    python "!PYTHON_SCRIPT!" ^
        --out_dir "!JSON_OUTPUT_DIR!" ^
        --summary_csv "!SUMMARY_CSV_PATH!" ^
        --seed !CURRENT_SEED! ^
        --Lx !Lx! ^
        --Ly !Ly! ^
        --km !km! ^
        --ki !ki! ^
        --N !N! ^
        --phi_target !PHI_TARGET! ^
        --gmin !GMIN! ^
        --boundary_margin !BOUNDARY_MARGIN! ^
        --a_min !A_MIN! ^
        --a_max !A_MAX! ^
        --b_min !B_MIN! ^
        --b_max !B_MAX! ^
        --theta_min !THETA_MIN! ^
        --theta_max !THETA_MAX!
    
    :: 可选：检查Python脚本是否成功执行
    IF !ERRORLEVEL! NEQ 0 (
        ECHO 警告: 样本 %%i 生成失败 ^(错误代码: !ERRORLEVEL!^)
    )
)


ECHO.
ECHO ===================================================
ECHO 批量生成完成!
ECHO.
ECHO - 总计生成了 !NUM_SAMPLES! 个样本。
ECHO - JSON文件保存在: "!JSON_OUTPUT_DIR!"
ECHO - 汇总数据保存在: "!SUMMARY_CSV_PATH!"
ECHO ===================================================
ECHO.

:: PAUSE命令会暂停执行，等待用户按任意键，以便查看最终的输出信息
PAUSE
