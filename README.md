# 人工智能协会二面实战项目

## 项目简介
本项目为人工智能协会二面考核实战任务实现。整合了基于大语言模型的智能聊天助手与 YOLOv8 实时目标检测功能，提供统一的 Web 交互界面。

## 功能特性
- **智能聊天助手**：对接 LM Studio 本地大模型，支持天气、时间等工具调用（意图前置拦截）。
- **YOLO 目标检测**：集成 YOLOv8n 模型，支持摄像头调用、单帧检测与连续检测。
- **Web 服务**：基于 Flask 构建，提供前后端联调界面与 API 代理转发。

## 项目结构
text
D:\ai-project\
├── app.py # Flask 主服务
├── agent.py # 智能体逻辑
├── train_yolo.py # YOLO 训练脚本
├── yolo_realtime.py # YOLO 独立测试脚本
├── requirements.txt # 依赖清单
├── templates/ # 前端 HTML 模板
│ ├── index.html
│ └── templatesyolo.html
├── runs/ # 训练权重与输出（已忽略）
└── LOG.md # 排错日志
## 环境配置
1. 克隆仓库并进入目录：
powershell
git clone <your-repo-url>
cd D:\ai-project
2. 创建虚拟环境并安装依赖：
powershell
python -m venv .venv
..venv\Scripts\activate
pip install -r requirements.txt
3. 安装 PyTorch（需匹配本地 CUDA 版本以启用 GPU）：
powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
4. 启动 LM Studio，加载大模型并开启 Server 模式（默认端口 5001）。

## 运行步骤
1. 启动 Flask 主服务：
powershell
python app.py
2. 访问以下地址：
- 聊天助手：`http://localhost:5000`
- YOLO 检测：`http://localhost:5000/yolo`
- 健康检查：`http://localhost:5000/api/health`

## API 说明
- `GET /`：聊天助手主页
- `GET /yolo`：YOLO 检测演示页
- `POST /api/chat`：聊天接口，转发请求至 agent.py
- `GET /api/health`：服务健康检查

## AI 使用说明
本项目在开发过程中合理使用 AI 辅助工具：
- **环境排错**：PyTorch CUDA 版本安装、opencv 兼容性问题的排查。
- **代码生成**：Flask 路由结构、前端 yolo.html 页面、训练脚本参数配置的初始框架。
- **文档撰写**：README.md、LOG.md 的初稿整理。
- **思路设计**：工具调用前置拦截方案、模型懒加载策略、显存管理方案均为自主设计，AI 辅助实现细节。
