@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo HATA: Once setup.bat dosyasini calistir.
    pause
    exit /b 1
)

call ".venv\Scripts\python.exe" -m streamlit run app.py
pause
