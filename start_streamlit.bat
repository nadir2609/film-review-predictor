@echo off
echo ============================================
echo IMDB Sentiment Analysis - Starting Web App
echo ============================================
cd /d "%~dp0"
call env\Scripts\activate.bat
echo Starting Streamlit on http://localhost:8501
streamlit run app.py
pause
