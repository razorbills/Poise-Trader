@echo off
echo.
echo ======================================================================
echo           POISE TRADER - WEB DASHBOARD LAUNCHER
echo ======================================================================
echo.
echo Starting bot with web dashboard...
echo.
echo Dashboard will open at: http://localhost:5000
echo.
echo Controls:
echo   - Click "Aggressive" or "Normal" to select mode
echo   - Click "Start Trading" to begin
echo   - Click "Stop Trading" to pause
echo   - Monitor real-time stats and charts
echo.
echo This window shows LOGS ONLY
echo All controls are in the web dashboard
echo.
echo ======================================================================
echo.

python micro_trading_bot.py

pause
