# 人工智能协会创智部二面实战项目

> 本地大模型聊天助手 + YOLO 实时目标检测 + 华小牛 Harness 自动化框架

## 📖 项目简介

本项目为人工智能协会创智部二面考核实战任务实现，包含四大模块：

| 模块 | 说明 |
|------|------|
| 🤖 智能聊天助手 | 对接 LM Studio 本地大模型（Qwen2.5-7B），支持天气查询、时间获取等工具调用 |
| 🎯 YOLO 目标检测 | 基于 YOLOv8n 的实时目标检测，支持 Web 端摄像头调用与标注展示 |
| 🐮 华小牛 Harness | 参考 pi-agent 架构，实现"规划-执行-记忆"自动化实验闭环 |
| 📝 工程规范 | Git 版本控制、`.env` 密钥管理、LOG.md 工程日志、AI 使用声明 |

所有功能均运行在本地，无需联网（天气 API 除外），保护数据隐私。

---

## 📁 项目结构
.
├── app.py # Flask 主服务（聊天 + YOLO 接口）
├── agent.py # 智能体服务（意图识别 + 工具调用 + 模型调度）
├── train_yolo.py # YOLO 训练脚本（支持命令行参数）
├── yolo_realtime.py # YOLO 独立推理脚本
├── harness.py # 🐮 华小牛 Harness 主程序
├── memory.py # Harness 记忆模块
├── tools_registry.py # Harness 工具库
├── requirements.txt # Python 依赖清单
├── .env # 环境变量（高德 API Key，不入库）
├── .gitignore # Git 排除规则
├── README.md # 本文件
├── LOG.md # 工程日志
├── templates/
│ ├── index.html # 聊天助手页面
│ └── yolo.html # YOLO 检测页面
├── runs/ # 训练产出（不入库）
│ └── detect/
│ └── exp_yolov8n/
│ └── weights/
│ └── best.pt
├── harness_memory/ # Harness 执行日志（不入库）
└── .venv/ # Python 虚拟环境（不入库）
---

## 🛠️ 环境配置

### 1. 克隆项目
powershell
git clone https://github.com/zilingyin/first.git
cd first
### 2. 创建并激活虚拟环境
powershell
python -m venv .venv
.\.venv\Scripts\activate
### 3. 安装依赖
powershell
pip install -r requirements.txt
核心依赖：
- flask / flask-cors
- requests / python-dotenv
- ultralytics / opencv-python / numpy
- torch / torchvision（需匹配 CUDA 版本）

### 4. 安装 GPU 版 PyTorch
powershell
先查看 CUDA 版本
nvidia-smi
根据 CUDA 版本安装（示例为 CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
### 5. 配置环境变量

创建 `.env` 文件（与 `app.py` 同级）：
env
AMAP_API_KEY=你的高德天气API_Key
获取方式：登录 [高德开放平台](https://lbs.amap.com/) → 控制台 → 创建应用 → 获取 Web 服务 Key。

### 6. 启动 LM Studio

- 打开 LM Studio，加载模型（推荐 Qwen2.5-7B-Instruct）
- 开启 Local Server（默认端口 1234）
- 确认模型已加载完成

---

## 🚀 运行指南

### 任务一：聊天助手
powershell
终端 1：启动智能体服务
python agent.py
终端 2：启动 Web 服务
python app.py
浏览器访问
http://localhost:5000
### 任务二：YOLO 检测
powershell
方式一：独立脚本测试
python yolo_realtime.py
按 q 退出，按 s 保存当前帧
方式二：Web 端检测
确保 app.py 已启动，浏览器访问
http://localhost:5000/yolo
点击"开启摄像头" → "检测一帧" / "连续检测"
### 任务四：华小牛 Harness
powershell
前提：LM Studio 已启动（端口 1234），app.py 已启动（端口 5000）
cd D:\ai-project
..venv\Scripts\activate
启动 Harness
python harness.py
输入目标，如："自动完成YOLOv8n的训练并生成日志"
**一键启动脚本**（可选）：创建 `start_harness.bat`
bat
@echo off
cd /d D:\ai-project
call .venv\Scripts\activate
start "Flask" cmd /k "python app.py"
timeout /t 3
python harness.py
pause
> 双击即可自动启动 Flask + Harness（LM Studio 需手动开启 Server）。

---

## 🐮 华小牛 Harness 详解

### 架构设计

参考 pi-agent 架构，实现自动化实验闭环：
人类目标（自然语言）
↓
┌───────────────────────────────────────┐
│ Planner（规划器） │
│ 调用 LLM 将目标拆解为步骤链表 │
└───────────────────────────────────────┘
↓
┌───────────────────────────────────────┐
│ Execution Loop（执行循环） │
│ 逐步调用工具 → 更新状态 → 推进步骤 │
└───────────────────────────────────────┘
↓
┌───────────────────────────────────────┐
│ Tools Registry（工具库） │
│ ├── check_env（环境检查） │
│ ├── run_yolo_train（训练） │
│ ├── run_yolo_detect（检测） │
│ ├── write_log（写日志） │
│ └── call_llm（调用大模型） │
└───────────────────────────────────────┘
↓
┌───────────────────────────────────────┐
│ Memory（记忆模块） │
│ 任务状态持久化 + 执行日志追加 │
└───────────────────────────────────────┘
### 工具清单

| 工具名 | 参数 | 功能 |
|--------|------|------|
| `check_env` | 无 | 检查 GPU、PyTorch、CUDA、权重文件 |
| `run_yolo_train` | `data_yaml`, `epochs`, `batch`, `imgsz` | 启动 YOLO 训练 |
| `run_yolo_detect` | `source`, `conf` | 对图片做目标检测 |
| `write_log` | `content`, `filename` | 追加日志到文件 |
| `call_llm` | `prompt`, `system` | 调用本地大模型推理 |

### 输入目标示例
✅ 推荐输入：
"自动完成YOLOv8n的训练并生成日志"
"检查环境并训练3轮"
"训练模型并把结果写入LOG.md"
⚠️ 效果一般：
"帮我写首诗"（工具不匹配）
❌ 不支持：
多轮对话、联网搜索、打开浏览器等（工具库无对应工具）
### 设计理念

> Harness 不是通用 Agent，而是**领域限定**的自动化框架。它的工具库围绕 YOLO 视觉实验设计，能自主完成"环境检查 → 训练 → 检测 → 日志回写"的闭环。限定工具集 + 明确边界，才是可落地的 Agent 设计。

---

## 📊 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | yolov8n.pt | 最小最快，适配 8GB 显存 |
| 数据集 | coco128.yaml | 80 类通用目标 |
| epochs | 50 | 小数据集防过拟合 |
| batch | 8 | RTX 4070 实测上限 |
| imgsz | 640 | YOLO 标准尺寸 |
| patience | 10 | 早停策略 |
| amp | True | FP16 混合精度，提速省显存 |

训练产出：`runs/detect/exp_yolov8n/weights/best.pt`

---

## 🔌 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 聊天助手主页 |
| GET | `/yolo` | YOLO 检测页面 |
| POST | `/api/chat` | 聊天接口（转发至 agent.py） |
| GET | `/api/health` | 健康检查 |
| POST | `/api/yolo/detect` | YOLO 检测接口 |

### YOLO 检测接口示例
json
// 请求
POST /api/yolo/detect
{
"image": "data:image/jpeg;base64,/9j/..."
}
// 响应
{
"detections": [
{"label": "person", "confidence": 0.92, "bbox": [10, 20, 200, 300]}
],
"image": "data:image/jpeg;base64,/9j/...",
"count": 1
}
---

## 🏗️ 核心设计亮点

### 1. 意图前置拦截（任务一）
不把所有请求都丢给大模型，先用关键词匹配判断意图。命中"天气""时间"等直接调工具，其余才走大模型。
- **延迟低**：工具调用毫秒级
- **可控性强**：参数结构化，不会"瞎编"

### 2. 模型懒加载 + FP16（任务二）
- 只在第一次请求时加载 YOLO，避免和 LLM 争抢显存
- 推理用 FP16 半精度，速度提升约 30%，显存占用降至约 1GB

### 3. 规划-执行-记忆闭环（任务四）
- **Planner**：LLM 将自然语言目标拆解为步骤链表
- **Executor**：逐步调用工具，更新状态
- **Memory**：任务状态持久化到 JSON，执行日志追加到文件
- **容错**：某步失败不中断，继续执行下一步

---

## 🐛 踩坑记录

| 问题 | 原因 | 解决 |
|------|------|------|
| PyTorch 装成 CPU 版 | pip 默认装 CPU 版 | 用 `--index-url` 指定 CUDA 版本重装 |
| 权重路径嵌套两层 | `project` 参数重复拼接 | 修正为 `runs/detect` + `exp_yolov8n` |
| 访问 `/yolo` 报 500 | 模板文件名与代码不一致 | 重命名为 `yolo.html` + 开 `debug=True` |
| 直接双击 HTML 报 CORS | `file://` 协议无法发 HTTP | 必须走 `localhost:5000` |
| 显存被 LM Studio 占满 | 7B 模型占约 6GB | 训练时关 LM Studio / 用 FP16 |
| `UnicodeDecodeError` GBK | Windows 默认 GBK 解码 YOLO 输出 | `subprocess` 加 `encoding='utf-8'` |
| `NameError: load_state` | 未从 memory 导入 | 顶部明确 `from memory import load_state` |
| `NoneType` 报错 | 工具函数异常返回 None | 所有工具确保返回 dict + 防御判断 |

---

## 🐛 常见问题

### Q：启动后访问页面报 404？
确认访问的是 `http://localhost:5000/yolo`（HTTP 服务），不是直接双击打开 HTML 文件。

### Q：YOLO 检测报 500 错误？
1. 开启 `debug=True` 查看详细报错
2. 确认模板文件名与 `render_template()` 参数一致
3. 检查权重文件路径是否存在

### Q：显存不足？
- 训练时关闭 LM Studio
- 推理使用 FP16（`half=True`）
- 降低 batch size 至 4

### Q：聊天无回复？
1. 确认 LM Studio 已启动且 Server 端口为 1234
2. 确认已加载模型
3. 访问 `/api/health` 查看 `agent_online` 状态

### Q：Harness 报连接拒绝？
1. 确认 LM Studio Server 已启动（端口 1234）
2. 确认 `app.py` 已运行（端口 5000）
3. 检查 `.env` 文件是否存在

---

## 📝 AI 使用说明

本项目在开发过程中合理使用 AI 辅助工具：

- **环境排错**：PyTorch CUDA 版本安装、opencv 兼容性问题的排查
- **代码框架**：Flask 路由结构、前端 yolo.html 页面、训练脚本参数配置的初始框架
- **文档撰写**：README.md、LOG.md 的初稿整理
- **架构设计**：意图前置拦截方案、模型懒加载策略、Harness 规划-执行-记忆闭环设计

---

*最后更新：2026-10-21*
