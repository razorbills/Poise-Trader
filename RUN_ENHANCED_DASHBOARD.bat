@echo off
cls
color 0B
echo.
echo  =====================================================
echo                   POISE TRADER
echo            ENHANCED DASHBOARD WITH BOT
echo  =====================================================
echo.
echo  Starting trading bot with full dashboard features...
echo.

:: Kill any existing processes
echo  [1] Cleaning up old processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

:: Check Python
echo  [2] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found! Please install Python 3.8+
    pause
    exit
)
echo      Python OK

:: Check dependencies
echo  [3] Checking dependencies...
pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo      Installing Flask...
    pip install flask flask-socketio >nul
)
echo      Dependencies OK

echo.
echo  [4] Launching Enhanced Dashboard System...
echo  =====================================================
echo.

:: Run the bot with dashboard
python run_bot_with_simple_dashboard.py

:: If script ends, pause
echo.
echo  =====================================================
echo  Trading system stopped.
echo  =====================================================
echo.
pause
