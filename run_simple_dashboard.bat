@echo off
echo ========================================
echo    SIMPLE TRADING CONTROL PANEL
echo ========================================
echo.

:: Start the simple dashboard server
echo Starting Simple Dashboard...
start cmd /k "python simple_dashboard_server.py"

:: Wait a moment for server to start
timeout /t 3 /nobreak >nul

:: Open the dashboard in browser
echo Opening dashboard in browser...
start http://localhost:5000

echo.
echo ========================================
echo    DASHBOARD READY!
echo ========================================
echo.
echo Dashboard opened in your browser
echo URL: http://localhost:5000
echo.
echo NOTE: You need to run your trading bot
echo separately for the controls to work!
echo.
pause
