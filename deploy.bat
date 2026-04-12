@echo off
REM Windows deployment script for Salevora
REM Run this as Administrator

setlocal enabledelayedexpansion

echo ================================================== 
echo SALEVORA WINDOWS DEPLOYMENT SCRIPT
echo ==================================================

REM Check for admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Please run this script as Administrator
    pause
    exit /b 1
)

REM Configuration
set APP_DIR=C:\Salevora
set PYTHON_EXE=python
set PORT=8000

echo [1/5] Creating application directory...
if not exist "%APP_DIR%" (
    mkdir "%APP_DIR%"
    echo Created directory: %APP_DIR%
)

echo [2/5] Creating virtual environment...
cd /d "%APP_DIR%"
if not exist "venv" (
    %PYTHON_EXE% -m venv venv
    echo Virtual environment created
)

echo [3/5] Installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn

echo [4/5] Creating startup script...
(
    echo @echo off
    echo cd /d "%APP_DIR%"
    echo call venv\Scripts\activate.bat
    echo python api.py --host 0.0.0.0 --port %PORT%
) > "%APP_DIR%\start.bat"
echo Startup script created: %APP_DIR%\start.bat

echo [5/5] Creating Windows Task Scheduler entry...
schtasks /create /tn "Salevora API" /tr "%APP_DIR%\start.bat" ^
    /sc onstart /rl highest /f

echo.
echo ==================================================
echo DEPLOYMENT SUCCESSFUL!
echo ==================================================
echo.
echo API URL: http://localhost:%PORT%
echo API Docs: http://localhost:%PORT%/docs
echo.
echo To start manually: %APP_DIR%\start.bat
echo To check scheduled task: schtasks /query /tn "Salevora API"
echo.
echo For production deployment, use Docker or Windows Service.
echo See DEPLOYMENT_GUIDE.md for details.
echo.
pause
