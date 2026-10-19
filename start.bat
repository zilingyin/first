@echo off
chcp 65001 >nul
cd /d D:\ai-project
call .venv\Scripts\activate

echo ========================================
echo   AI 智能助手 - 一键启动
echo ========================================
echo.

start "Agent Service (Port 5001)" cmd /k "cd /d D:\ai-project && call .venv\Scripts\activate && python agent.py"
start "Web App (Port 5000)" cmd /k "cd /d D:\ai-project && call .venv\Scripts\activate && python app.py"

echo ✅ 两个服务已在新窗口中启动
echo.
echo    前端界面：http://localhost:5000
echo    智能体API：http://localhost:5001
echo.
echo ⚠️  请确保 LM Studio 已加载模型并开启 Server（端口 1234）
echo.
pause