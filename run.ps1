Write-Host "Starting ClipMind AI Server..." -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
