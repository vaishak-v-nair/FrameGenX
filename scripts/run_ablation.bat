@echo off
REM T9: Overnight ablation — ONE PROCESS PER CONFIG (eng review OV#3).
REM Process isolation: a crash in one config cannot kill the others, and the
REM LoRA pre-offload hook fires correctly on each fresh pipe construction.
REM Resume is DB-keyed on completed (config_name, prompt, seed) rows, so
REM infrastructure errors remain retryable on the next run.
REM
REM Pre-launch (manual, once): powercfg /change standby-timeout-ac 0
REM Next morning: powercfg /change standby-timeout-ac 30  (or your default)

cd /d "%~dp0.."
REM 15-prompt cut ARMED (gate projected 14h at cold-start rate; balanced
REM 10 object_count + 5 color_constraints). Rerun with --limit 0 later to
REM finish the full 30 — resume skips completed rows.
set LIMIT=15
set "FAILED=0"

echo === [1/4] baseline ===
venv\Scripts\python.exe benchmark.py --config baseline --limit %LIMIT%
if errorlevel 1 set "FAILED=1"
echo === [2/4] rag ===
venv\Scripts\python.exe benchmark.py --config rag --limit %LIMIT%
if errorlevel 1 set "FAILED=1"
echo === [3/4] rag_lora ===
venv\Scripts\python.exe benchmark.py --config rag_lora --limit %LIMIT%
if errorlevel 1 set "FAILED=1"
echo === [4/4] full ===
venv\Scripts\python.exe benchmark.py --config full --limit %LIMIT%
if errorlevel 1 set "FAILED=1"

echo === Rendering results ===
venv\Scripts\python.exe scripts\make_results.py
if errorlevel 1 set "FAILED=1"
if "%FAILED%"=="1" (
    echo === ABLATION INCOMPLETE: review errors and rerun ===
    exit /b 1
)
echo === ABLATION COMPLETE ===
