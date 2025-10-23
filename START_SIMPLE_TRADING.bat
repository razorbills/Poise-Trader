@echo off
cls
color 0A
echo.
echo  =====================================================
echo                  POISE TRADER
echo            SIMPLE CONTROL PANEL LAUNCHER
echo  =====================================================
echo.
echo  Starting your trading bot with simple controls...
echo.
echo  [1] Initializing systems...
timeout /t 2 /nobreak >nul

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Python not found!
    echo  Please install Python 3.8+ first.
    pause
    exit
)

echo  [2] Checking dependencies...
pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo      Installing Flask...
    pip install flask flask-socketio >nul 2>&1
)
echo      Dependencies OK

echo  [3] Launching Trading System...
echo.
echo  =====================================================
echo.

:: Run the bot with simple dashboard
python run_bot_with_simple_dashboard.py

:: If script ends, pause
echo.
echo  =====================================================
echo  Trading system stopped.
echo  =====================================================
echo.
pause
