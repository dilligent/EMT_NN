@ECHO OFF
TITLE Batch Structure Generator

:: �����ӳٱ���չ�����������ѭ����ʹ�ñ����ǳ���Ҫ
SETLOCAL EnableDelayedExpansion

:: ==========================================================================
:: ������ű��������������� generate_elliptic_composites.py
:: ==========================================================================
:: ����:
::   �˽ű�ͨ��ѭ����ε���Python�ű���Ϊÿ�ε�������һ��Ψһ��������ӣ�
::   �Ӷ�����������ͬ�ļ��νṹ������
::
:: �޸�˵��:
::   - �����ӳٱ���չ����ȷ��RANDOM��ÿ��ѭ���ж���������
::   - �޸��˱�������������
::
:: ʹ�÷���:
::   1. ���� .bat �ļ��� generate_elliptic_composites.py ����ͬһ���ļ����С�
::   2. �޸�����ġ���������������
::   3. ˫�����д� .bat �ļ���
:: ==========================================================================


:: --- ���������� ---

:: ����Python�ű����ļ���
SET "PYTHON_SCRIPT=generate_elliptic_composites.py"

:: ��������ļ�����Ŀ¼
SET "OUTPUT_BASE_DIR=generated_samples_bat"

:: ����Ҫ���ɵ���������
SET "NUM_SAMPLES=200"

:: --- Python�ű��������� ---
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


:: --- �ű�ִ���� (һ�������޸�) ---

ECHO.
ECHO ===================================================
ECHO      �������������� (CMD Batch Version - Fixed)
ECHO ===================================================
ECHO.

:: ���Python�ű��Ƿ����
IF NOT EXIST "!PYTHON_SCRIPT!" (
    ECHO ����: Python�ű� "!PYTHON_SCRIPT!" ������!
    ECHO ��ȷ�����������ļ���Python�ű���ͬһ��Ŀ¼�¡�
    PAUSE
    EXIT /B 1
)

:: ׼�����Ŀ¼
SET "JSON_OUTPUT_DIR=!OUTPUT_BASE_DIR!\json_files"
SET "SUMMARY_CSV_PATH=!OUTPUT_BASE_DIR!\samples_summary.csv"

ECHO ׼�����Ŀ¼: !JSON_OUTPUT_DIR!
IF NOT EXIST "!JSON_OUTPUT_DIR!" (
    MKDIR "!JSON_OUTPUT_DIR!"
)

:: ɾ���ɵĻ����ļ���ȷ��ÿ�����ж���ȫ�µĿ�ʼ
IF EXIST "!SUMMARY_CSV_PATH!" (
    DEL "!SUMMARY_CSV_PATH!"
    ECHO ��ɾ���ɵĻ���CSV�ļ���
)

ECHO.
ECHO ������ʼ���� %NUM_SAMPLES% ������...
ECHO.
timeout /t 3 >nul

:: ��ѭ��
:: FOR /L %%i IN (start, step, end) DO (command)
FOR /L %%i IN (1, 1, %NUM_SAMPLES%) DO (
    
    :: ʹ���ӳ�չ���﷨ !RANDOM! ��ȷ��ÿ��ѭ���������µ������
    SET "CURRENT_SEED=!RANDOM!"
    
    ECHO ���ڴ������� %%i / %NUM_SAMPLES% ^(ʹ������: !CURRENT_SEED!^)
    
    :: ����Python�ű����������в���
    :: ע��: CMD��û��--verbose�������ǿ���ͨ�����ض�������ﵽ����Ч��
    :: ʹ���ӳ�չ���﷨ !����! ����ȡ���б����ĵ�ǰֵ
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
    
    :: ��ѡ�����Python�ű��Ƿ�ɹ�ִ��
    IF !ERRORLEVEL! NEQ 0 (
        ECHO ����: ���� %%i ����ʧ�� ^(�������: !ERRORLEVEL!^)
    )
)


ECHO.
ECHO ===================================================
ECHO �����������!
ECHO.
ECHO - �ܼ������� !NUM_SAMPLES! ��������
ECHO - JSON�ļ�������: "!JSON_OUTPUT_DIR!"
ECHO - �������ݱ�����: "!SUMMARY_CSV_PATH!"
ECHO ===================================================
ECHO.

:: PAUSE�������ִͣ�У��ȴ��û�����������Ա�鿴���յ������Ϣ
PAUSE
