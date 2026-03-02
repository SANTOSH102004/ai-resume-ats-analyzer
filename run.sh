#!/bin/bash
# Run script for local and cloud deployment
# This script handles PORT environment variable for cloud platforms (Render, Streamlit Cloud, etc.)

# Get port from environment variable or use default
PORT=${PORT:-8501}

# Run Streamlit with the specified port
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
