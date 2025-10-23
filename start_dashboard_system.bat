@echo off
echo ===============================================
echo     POISE TRADER DASHBOARD STARTUP SYSTEM
echo ===============================================
echo.

:: Install Python dependencies if needed
echo [1/3] Checking Python dependencies...
pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Python dependencies...
    pip install Flask flask-socketio flask-cors python-socketio
) else (
    echo Python dependencies OK
)

:: Start backend server
echo.
echo [2/3] Starting Backend Server...
start /min cmd /k "cd /d %~dp0 && python dashboard_backend.py"
timeout /t 3 /nobreak >nul

:: Start frontend dev server
echo.
echo [3/3] Starting React Dashboard...
cd dashboard
start cmd /k "npm run dev"

echo.
echo ===============================================
echo     DASHBOARD SYSTEM STARTED SUCCESSFULLY!
echo ===============================================
echo.
echo Backend Server: http://localhost:5000
echo React Dashboard: http://localhost:5173
echo.
echo The dashboard will open in your browser automatically.
echo Press any key to close this window (servers will keep running).
echo.
pause >nul
