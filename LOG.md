# 项目工程日志


---

## 📌 项目概述

- **项目名称**：本地大模型聊天 + YOLO 目标检测 + 华小牛 Harness
- **开发者**：zilingyin
- **仓库地址**：https://github.com/zilingyin/first
- **AI 使用说明**：本项目在环境排错、代码框架生成、文档撰写阶段合理使用 AI 辅助，所有代码均经人工调试与修改。

---

## 📅 第一阶段：环境搭建（Day 1）

### ✅ 已完成
- 安装 Python 3.12，创建虚拟环境 `.venv`
- 安装核心依赖：torch、ultralytics、opencv-python、flask、flask-cors、requests
- 部署 LM Studio，加载 Qwen2.5-7B-Instruct 模型，开启 Server（端口 1234）

### ⚠️ 踩坑记录
**问题**：`torch.cuda.is_available()` 返回 False，YOLO 只能用 CPU。
**原因**：pip 默认安装的是 CPU 版本 PyTorch。
**解决**：
powershell
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
**耗时**：约 1 小时（反复卸载重装）。
**教训**：装 PyTorch 前一定要先确认 CUDA 版本，用 `nvidia-smi` 查看。

---

## 📅 第二阶段：任务一 · 智能聊天助手（Day 2-3）

### ✅ 已完成
- 搭建 Flask 后端 `app.py`（端口 5000），负责路由与前端渲染
- 搭建智能体服务 `agent.py`（端口 5001），负责意图识别 + 工具调用 + 模型调度
- 实现两个工具：
  - **天气查询**：接入高德天气 API，支持全国城市
  - **时间获取**：返回系统当前时间
- 前端聊天界面 `index.html`，支持对话历史 localStorage 持久化

### 🧠 架构设计亮点
**意图前置拦截**：不把所有请求都丢给大模型，而是先用关键词匹配判断意图。命中"天气""时间"等关键词直接调工具，其余才走大模型。好处：
- 延迟低（工具调用毫秒级，不用等模型推理）
- 可控性强（工具参数结构化，不会出现模型"瞎编"参数）

### ⚠️ 踩坑记录
**问题**：`AMAP_API_KEY` 写在代码里，push 到 GitHub 后暴露了密钥。
**解决**：改用 `.env` 文件 + `python-dotenv` 加载，并把 `.env` 加入 `.gitignore`。
**教训**：密钥永远不能进版本库。

---

## 📅 第三阶段：任务二 · YOLO 目标检测（Day 4-5）

### ✅ 已完成
- 完成 YOLOv8n 训练：COCO128 数据集，50 轮，batch=8，imgsz=640
- 编写 `train_yolo.py`，支持命令行参数，产出 `runs/detect/exp_yolov8n/weights/best.pt`
- 编写 `yolo_realtime.py`，独立脚本可弹出摄像头并实时检测
- 编写 `templates/yolo.html`，Web 端可开启摄像头、单帧检测、连续检测
- Flask 新增 `/api/yolo/detect` 接口，接收 base64 图片 → YOLO 推理 → 返回标注图 + 检测列表

### ⚠️ 踩坑记录

#### 坑 1：权重路径嵌套了两层
**现象**：代码里写 `runs/detect/runs/detect/exp_yolov8n/weights/best.pt`，实际不存在。
**原因**：`train_yolo.py` 里的 `project` 参数设成了 `runs/detect`，而 YOLO 训练本身就会往 `runs/detect/` 下输出，导致嵌套。
**解决**：把 `project` 改为 `runs/detect`，`name` 改为 `exp_yolov8n`，产出路径变为 `runs/detect/exp_yolov8n/weights/best.pt`。代码中加了候选路径列表自动查找。

#### 坑 2：访问 `/yolo` 报 500 Internal Server Error
**现象**：浏览器白屏，只显示 "500 Internal Server Error"。
**排查过程**：
1. 第一反应看终端 —— 没有报错输出（因为 debug=False）
2. 开启 `debug=True` 重启，浏览器显示详细错误：`TemplateNotFound: yolo.html`
3. 发现模板文件名叫 `templatesyolo.html`，但代码里写的是 `render_template("yolo.html")`
**解决**：把模板文件重命名为 `yolo.html`。
**教训**：
- 开发阶段一定要开 `debug=True`
- 文件名和代码引用必须严格一致，前后端联调先核对路由和文件名

#### 坑 3：直接双击打开 HTML 文件导致 CORS 报错
**现象**：F12 控制台报跨域错误，请求地址变成 `file:///D:/api/yolo/detect`。
**原因**：直接双击 HTML 文件，页面以 `file://` 协议运行，无法发起 HTTP 请求。
**解决**：必须通过 `http://localhost:5000/yolo` 访问。
**教训**：前端调后端接口，一定要走 HTTP 服务，不能本地文件直开。

#### 坑 4：显存被 LM Studio 占满
**现象**：启动 YOLO 检测时报显存不足。
**原因**：LM Studio 加载 7B 模型占用了约 6GB 显存，YOLO 训练/推理需要额外显存。
**解决**：
- 训练时关闭 LM Studio
- 推理时用 FP16 半精度（`half=True`），YOLOv8n 只需约 1GB 显存
- 采用模型懒加载，只在第一次请求时加载 YOLO，避免和 LLM 服务同时占用

### 🧠 性能优化
- **FP16 半精度推理**：速度提升约 30%
- **模型懒加载**：避免与 LLM 争抢显存
- **降低输入分辨率**：640×480 摄像头画面，FPS 从 25 提升到 35-40

---

## 📅 第四阶段：任务四 · 华小牛 Harness（Day 6-7）

### ✅ 已完成
参考 pi-agent 架构，实现自动化视觉实验闭环 Harness，包含 4 个模块：

| 模块 | 文件 | 职责 |
|------|------|------|
| 记忆模块 | `memory.py` | 任务状态持久化、执行日志追加 |
| 工具库 | `tools_registry.py` | 封装 YOLO 训练/检测、环境检查、日志写入、LLM 调用 |
| 规划器 | `harness.py` | 调用 LLM 将人类目标拆解为步骤链表 |
| 执行循环 | `harness.py` | 逐步执行工具，更新状态，直到任务完成 |

### 🎯 运行效果
输入目标："自动完成YOLOv8n的训练并生成日志"
Harness 自主完成：
1. ✅ 检查环境（检测到 RTX 4070、权重文件存在）
2. ✅ 调用 `train_yolo.py` 启动训练
3. ✅ 训练完成后写入日志

### ⚠️ 踩坑记录

#### 坑 5：Windows GBK 解码崩溃（UnicodeDecodeError）
**现象**：Harness 调用 `run_yolo_train` 工具时，终端报 `UnicodeDecodeError: 'gbk' codec can't decode byte 0x...`。
**原因**：`subprocess.run` 在 Windows 上默认用 GBK 解码子进程输出，而 YOLO 训练脚本输出含 UTF-8 编码的 emoji 进度条，GBK 无法解码。
**连锁反应**：解码崩溃 → 工具函数未返回字典 → `execute_step` 拿到 None → `'NoneType' object is not subscriptable` 二次报错。
**解决**：
python
result = subprocess.run(
cmd, capture_output=True, text=True,
encoding='utf-8', errors='ignore', # ← 关键修复
timeout=7200
)
**教训**：Windows 下 `subprocess` 处理含 emoji/中文的子进程输出，必须显式指定 `encoding='utf-8'`。

#### 坑 6：NameError: name 'load_state' is not defined
**现象**：Harness 执行完所有步骤后，打印最终状态时崩溃。
**原因**：`harness.py` 末尾调用了 `load_state()`，但只从 memory 导入了部分函数，漏掉了 `load_state`。
**解决**：顶部导入改为完整列表：
python
from memory import (
init_memory, add_task, update_task_status, save_plan,
get_current_plan, get_step_index, advance_step, log_execution,
load_state, # ← 补上
)
**教训**：导入时宁可多写几个名字，也不要用 `from memory import *`（会掩盖遗漏）。

#### 坑 7：工具函数返回 None 导致 NoneType 报错
**现象**：某个工具执行异常后，`execute_step` 里 `result.get("status")` 报 `'NoneType' object has no attribute 'get'`。
**原因**：工具函数在异常路径下没有 return，Python 函数默认返回 None。
**解决**（双层防御）：
1. `execute_tool` 里加类型检查：`if not isinstance(result, dict): return {"status": "failed", ...}`
2. 每个工具函数的 `except` 块都 `return {"status": "failed", "error": str(e)}`
3. `execute_step` 里加 `if not isinstance(result, dict)` 判断
**教训**：Agent 工具库是"高容错"场景，每个出口都必须有返回值。

#### 坑 8：训练时终端"卡住"不动
**现象**：Harness 启动训练后，终端长时间无输出，看起来像卡死。
**原因**：`subprocess.run(capture_output=True)` 捕获了训练脚本的全部输出，进度条不实时显示。
**解决**：这是正常现象，训练本身耗时较长。如需实时输出，可改用 `subprocess.Popen` 逐行读取 stdout。
**临时方案**：测试时用 `--epochs 3` 快速验证链路，正式训练再调 50。

### 🧠 架构思考
传统 Agent = 大模型 + 工具调用（被动响应）。
Harness = 规划器 + 工具库 + 记忆 + 执行循环（主动闭环）。
人类只需作为"启动子"输入一句目标，Harness 自主完成整个实验流程。

**设计取舍**：华小牛不是通用 Agent，而是领域限定的自动化框架。工具库围绕 YOLO 视觉实验设计，能自主完成"环境检查→训练→检测→日志回写"闭环。限定工具集 + 明确边界，才是可落地的 Agent 设计——而不是做一个"什么都能干但什么都干不好"的万能助手。

---
日期：2026-10-21

