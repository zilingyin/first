@echo off
cd /d D:\ai-project
call .venv\Scripts\activate
start "Flask" cmd /k "python app.py"
timeout /t 3
python harness.py
pause