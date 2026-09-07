@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo CareerLens - Foundry Local RAG Kurulumu
echo ==============================================
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo HATA: Python launcher ^(py^) bulunamadi.
    echo Python 3.11 veya daha yeni bir surum kurup tekrar dene.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Sanal ortam olusturuluyor...
    py -m venv .venv
    if errorlevel 1 goto :error
) else (
    echo [1/4] Sanal ortam zaten mevcut.
)

echo [2/4] pip guncelleniyor...
call ".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [3/4] Proje bagimliliklari kuruluyor...
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [4/4] Paketler kontrol ediliyor...
call ".venv\Scripts\python.exe" -c "import foundry_local_sdk, streamlit, pypdf, docx, numpy; print('Paketler basariyla kuruldu.')"
if errorlevel 1 goto :error

echo.
echo KURULUM TAMAMLANDI.
echo Siradaki adim: run_test.bat, ardindan run_app.bat
pause
exit /b 0

:error
echo.
echo KURULUM BASARISIZ.
pause
exit /b 1
