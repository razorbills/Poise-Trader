@echo off
echo.
echo ╔═══════════════════════════════════════════════════════════╗
echo ║  🆓 PREPARE FILES FOR FREE CLOUD HOSTING                 ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.

REM Create a deployment package folder
set PACKAGE_DIR=cloud_deployment_package
echo 📦 Creating deployment package...
echo.

REM Create directory
if exist "%PACKAGE_DIR%" rmdir /s /q "%PACKAGE_DIR%"
mkdir "%PACKAGE_DIR%"
mkdir "%PACKAGE_DIR%\ai_enhancements"
mkdir "%PACKAGE_DIR%\core"
mkdir "%PACKAGE_DIR%\core\feeds"
mkdir "%PACKAGE_DIR%\core\execution"
mkdir "%PACKAGE_DIR%\core\brain"
mkdir "%PACKAGE_DIR%\config"

echo ✅ Created package directory
echo.

REM Copy main files
echo 📋 Copying main files...
copy "micro_trading_bot.py" "%PACKAGE_DIR%\" >nul 2>&1
copy "cloud_launcher_free.py" "%PACKAGE_DIR%\" >nul 2>&1
copy "requirements_free_hosting.txt" "%PACKAGE_DIR%\requirements.txt" >nul 2>&1
copy ".env" "%PACKAGE_DIR%\" >nul 2>&1
echo    ✓ Main files copied
echo.

REM Copy AI enhancements
echo 🧠 Copying AI enhancements...
xcopy "ai_enhancements\*.py" "%PACKAGE_DIR%\ai_enhancements\" /Y >nul 2>&1
echo    ✓ AI files copied
echo.

REM Copy core modules
echo ⚙️  Copying core modules...
xcopy "core\*.py" "%PACKAGE_DIR%\core\" /Y >nul 2>&1
xcopy "core\feeds\*.py" "%PACKAGE_DIR%\core\feeds\" /Y >nul 2>&1
xcopy "core\execution\*.py" "%PACKAGE_DIR%\core\execution\" /Y >nul 2>&1
xcopy "core\brain\*.py" "%PACKAGE_DIR%\core\brain\" /Y >nul 2>&1
echo    ✓ Core files copied
echo.

REM Copy config
echo 📝 Copying config files...
xcopy "config\*.yaml" "%PACKAGE_DIR%\config\" /Y >nul 2>&1
xcopy "config\*.json" "%PACKAGE_DIR%\config\" /Y >nul 2>&1
echo    ✓ Config files copied
echo.

REM Create README for the package
echo 📄 Creating deployment README...
(
echo # DEPLOYMENT PACKAGE FOR FREE CLOUD HOSTING
echo.
echo This package contains everything needed to deploy your bot to:
echo - PythonAnywhere
echo - Replit
echo - Render
echo - Other free hosting platforms
echo.
echo ## Files Included:
echo - cloud_launcher_free.py ^(main launcher^)
echo - micro_trading_bot.py ^(core bot^)
echo - requirements.txt ^(dependencies^)
echo - .env ^(your API keys^)
echo - ai_enhancements/ ^(AI modules^)
echo - core/ ^(core trading modules^)
echo - config/ ^(configuration files^)
echo.
echo ## Quick Start:
echo 1. Upload all files to your hosting platform
echo 2. Install: pip install -r requirements.txt
echo 3. Run: python cloud_launcher_free.py
echo.
echo ## See QUICK_START_FREE_HOSTING.txt for detailed instructions!
) > "%PACKAGE_DIR%\README.md"

echo    ✓ README created
echo.

echo ═══════════════════════════════════════════════════════════
echo 🎉 PACKAGE READY!
echo ═══════════════════════════════════════════════════════════
echo.
echo 📂 Location: %CD%\%PACKAGE_DIR%
echo.
echo 📦 Package Contents:
dir /b "%PACKAGE_DIR%"
echo.
echo ═══════════════════════════════════════════════════════════
echo 📤 NEXT STEPS:
echo ═══════════════════════════════════════════════════════════
echo.
echo Option A - Direct Upload:
echo    1. Go to PythonAnywhere.com
echo    2. Upload all files from '%PACKAGE_DIR%' folder
echo.
echo Option B - GitHub:
echo    1. Create GitHub repository
echo    2. Upload files from '%PACKAGE_DIR%' folder
echo    3. Clone in PythonAnywhere
echo.
echo 💡 See QUICK_START_FREE_HOSTING.txt for step-by-step guide!
echo.

REM Open the folder
explorer "%PACKAGE_DIR%"

echo 📂 Package folder opened!
echo.
echo ✅ Ready to deploy to free cloud hosting!
echo.
pause
