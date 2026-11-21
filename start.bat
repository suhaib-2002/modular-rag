@echo off
REM ============================================================
REM  Modular RAG - Double click to run
REM  Creates venv (if missing), installs packages,
REM  then starts the server.
REM ============================================================

cd /d "%~dp0"

echo.
echo ============================================
echo   Modular RAG Server Starting...
echo ============================================
echo.

REM ---- Step 1: Create venv if it does not exist ----
if not exist "venv\Scripts\activate.bat" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create venv. Is Python installed? Check "python --version".
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment already exists.
)

REM ---- Step 2: Activate venv ----
call "venv\Scripts\activate.bat"

REM ---- Step 3: Install requirements ----
echo [2/3] Installing/verifying packages... (first run will take time)
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Packages failed to install. Check your internet connection.
    pause
    exit /b 1
)

REM ---- Step 4: Run the server ----
echo.
echo [3/3] Starting the server...
echo Open in browser:  http://localhost:8000/docs
echo To stop, press CTRL + C in this window.
echo.
uvicorn modular_rag:app --reload --host 0.0.0.0 --port 8000

pause
