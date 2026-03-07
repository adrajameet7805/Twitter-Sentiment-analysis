@echo off
echo ========================================
echo   STREAMLIT HTML FIX - AUTO RESTART
echo ========================================
echo.

echo [1/4] Clearing Streamlit cache...
streamlit cache clear
echo     ✓ Cache cleared
echo.

echo [2/4] Stopping any running Streamlit servers...
taskkill /F /IM streamlit.exe 2>nul
timeout /t 2 /nobreak >nul
echo     ✓ Servers stopped
echo.

echo [3/4] Starting fresh Streamlit server...
echo     Opening http://localhost:8501
echo.
start "" streamlit run app.py
timeout /t 5 /nobreak >nul
echo     ✓ Server started
echo.

echo [4/4] MANUAL STEPS REQUIRED:
echo     1. Clear browser cache (Ctrl+Shift+Delete)
echo     2. Hard refresh browser (Ctrl+F5)
echo.

echo ========================================
echo   Server is running!
echo   URL: http://localhost:8501
echo ========================================
echo.
echo Press any key to keep this window open...
pause >nul
