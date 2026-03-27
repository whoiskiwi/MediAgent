#!/bin/bash
# Start FastAPI and Streamlit in the same container
uvicorn main:app --host 0.0.0.0 --port 8000 &
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
