@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo HATA: Once setup.bat dosyasini calistir.
    pause
    exit /b 1
)

echo ==============================================
echo CareerLens - Cekirdek + Foundry Local Testleri
echo ==============================================
echo.
echo [1/2] Cekirdek RAG ve guvenlik testleri calistiriliyor...
call ".venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 goto :error

echo.
echo [2/2] Foundry Local model testi calistiriliyor...
echo Ilk calistirmada qwen2.5-0.5b CPU modeli indirilebilir ve bu biraz surebilir.
echo CUDA/TensorRT ile ilgili opsiyonel uyarilar gorunebilir; sohbet modeli CPU varyantina sabitlenmistir.
echo.
call ".venv\Scripts\python.exe" test_foundry.py
if errorlevel 1 goto :error

echo.
echo TUM TESTLER BASARILI.
pause
exit /b 0

:error
echo.
echo TEST BASARISIZ.
pause
exit /b 1
