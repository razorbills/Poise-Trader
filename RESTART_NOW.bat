@echo off
echo.
echo ========================================
echo   KILLING OLD BOT AND RESTARTING
echo ========================================
echo.

taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Starting bot with LATEST FIXES...
echo.

python micro_trading_bot.py

pause
