@echo off
title FrameGenX Demo
REM FrameGenX one-click demo. Double-click this file.
REM ONE window: the server runs right here. Close this window to stop.
REM The browser opens by itself as soon as the server is ready.
REM %~dp0 = the folder this .bat lives in, so it works from any clone path.

cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo [ERROR] venv not found. Do the one-time setup in README.md first:
    echo     python -m venv venv
    echo     venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if exist E:\hf_cache set HF_HOME=E:\hf_cache
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Already running? Just open the page and leave.
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8000/docs | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 (
    echo Server is already running - opening the page.
    start http://127.0.0.1:8000/app
    exit /b 0
)

if not exist frontend\dist (
    echo Building the web UI first ^(one-time, ~1 min, needs Node.js^)...
    pushd frontend
    call npm install
    call npm run build
    popd
)

echo.
echo  =================================================
echo   FrameGenX demo
echo   1. The server starts in THIS window ^(log below^)
echo   2. The browser opens BY ITSELF when it is ready
echo   3. CLOSE THIS WINDOW to stop the demo
echo  =================================================
echo.

REM Background helper: poll until the server answers, then open the browser.
REM If the server never comes up, say so instead of opening a dead page.
start "" /min powershell -NoProfile -Command "$ok=$false; for($i=0;$i -lt 90;$i++){ try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8000/docs | Out-Null; $ok=$true; break } catch { Start-Sleep 1 } }; if($ok){ Start-Process http://127.0.0.1:8000/app } else { Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('The FrameGenX server did not start. Look for the error in the FrameGenX Demo window.','FrameGenX') | Out-Null }"

venv\Scripts\python.exe -m src.api

REM If we get here the server stopped (crash or Ctrl+C). Keep the window
REM open so the error above stays readable.
echo.
echo =================================================
echo  The server stopped. If there is an error above,
echo  screenshot it. Otherwise just close this window.
echo =================================================
pause
