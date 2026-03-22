@echo off
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
REM Nếu dùng PowerShell, chạy: .\.venv\Scripts\Activate.ps1 trước
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
pause
