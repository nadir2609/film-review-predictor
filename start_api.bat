@echo off
echo ============================================
echo IMDB Sentiment Analysis - Starting API
echo ============================================
cd /d "%~dp0"
call env\Scripts\activate.bat
echo Starting FastAPI server on http://localhost:8000
python api.py
pause
