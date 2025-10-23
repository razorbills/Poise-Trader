@echo off
echo ================================================
echo    POISE TRADER - REAL DATA DASHBOARD SYSTEM
echo ================================================
echo.

:: Kill any existing Python processes running dashboard
echo [1/4] Stopping old processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *dashboard_backend*" >nul 2>&1
taskkill /F /IM node.exe /FI "WINDOWTITLE eq *npm*" >nul 2>&1
timeout /t 2 /nobreak >nul

:: Install dependencies if needed
echo [2/4] Checking dependencies...
pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Python dependencies...
    pip install flask flask-socketio
)

:: Start the combined bot + dashboard system
echo.
echo [3/4] Starting Trading Bot with Dashboard Integration...
echo.
start cmd /k "cd /d %~dp0 && python run_bot_with_dashboard.py"
timeout /t 5 /nobreak >nul

:: Start React dashboard
echo.
echo [4/4] Starting React Dashboard...
cd dashboard
start cmd /k "npm run dev"

echo.
echo ================================================
echo       REAL TRADING DASHBOARD STARTED!
echo ================================================
echo.
echo Dashboard URL: http://localhost:5173
echo.
echo IMPORTANT:
echo - The bot is now connected to the dashboard
echo - You will see REAL trading data
echo - Click "Start Trading" in the dashboard to begin
echo.
echo Press any key to close this window...
pause >nul
