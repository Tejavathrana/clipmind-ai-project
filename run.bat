@echo off
echo Starting ClipMind AI Server...
".venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
pause
