@echo off
REM Run script for local and cloud deployment
REM This script handles PORT environment variable for cloud platforms (Render, Streamlit Cloud, etc.)

REM Get port from environment variable or use default
if "%PORT%"=="" set PORT=8501

REM Run Streamlit with the specified port
streamlit run app.py --server.port %PORT% --server.address 0.0.0.0
