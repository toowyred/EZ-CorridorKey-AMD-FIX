@echo off
REM Corridor Key Launcher
set "LINUX_USER=corridor"
set "LINUX_IP=10.10.10.109"

REM Use full path to python in venv and script
set "REMOTE_PYTHON=/home/corridor/CorridorKey/venv/bin/python"
set "REMOTE_SCRIPT=/home/corridor/CorridorKey/clip_manager.py"

REM SAFETY CHECK: Ensure a folder was dragged onto the script
if "%~1"=="" (
    echo [ERROR] No target folder provided.
    echo.
    echo USAGE: 
    echo Please DRAG AND DROP a folder onto this script to process it.
    echo Do not double-click this script directly.
    echo.
    pause
    exit /b
)

REM Folder dragged? Use it as the target path.
set "WIN_PATH=%~1"

REM Strip trailing slash if present
if "%WIN_PATH:~-1%"=="\" set "WIN_PATH=%WIN_PATH:~0,-1%"

echo Connecting to Corridor Key on %LINUX_IP%...
echo Target: "%WIN_PATH%"

REM Connect via SSH and run the python script with the windows path
ssh -t %LINUX_USER%@%LINUX_IP% "%REMOTE_PYTHON% %REMOTE_SCRIPT% --action wizard --win_path '%WIN_PATH%'"

pause
