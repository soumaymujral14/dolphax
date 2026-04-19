@echo off
echo Starting SmartClaw...
echo Installing dependencies...
cd backend
pip install -r requirements.txt
echo Starting server on http://localhost:8000
uvicorn main:app --reload --port 8000
