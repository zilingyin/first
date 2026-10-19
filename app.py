"""
app.py —— Web 应用层
===================
职责：用户交互、会话管理、前端持久化接口、YOLO 检测接口
端口：5000
依赖：flask, flask-cors, requests, opencv-python, ultralytics, numpy

启动：python app.py
"""

import os
import sys
import json
import time
import base64
import logging
import requests

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ---------- YOLO 相关（可选，未安装时不阻塞聊天功能） ----------
YOLO_AVAILABLE = False
try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO 依赖已加载")
except ImportError as e:
    print(f"⚠️  YOLO 相关依赖未安装，/yolo 功能将不可用（聊天功能正常）: {e}")

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------- Flask 应用 ----------
app = Flask(__name__)
CORS(app)  # 允许跨域（前端 5000 调用后端 5001 必需）

# ---------- 配置项 ----------
AGENT_URL = "http://localhost:5001"   # 智能体服务地址

# YOLO 权重路径（自动查找，优先自定义权重）
_YOLO_CANDIDATES = [
    r"D:\ai-project\runs\detect\exp_yolov8n\weights\best.pt",           # 自定义训练权重
    r"D:\ai-project\runs\detect\runs\detect\exp_yolov8n\weights\best.pt",  # 嵌套目录（兼容）
    r"D:\ai-project\yolov8n.pt",                                        # 预训练权重
]
YOLO_WEIGHTS = None
if YOLO_AVAILABLE:
    for _p in _YOLO_CANDIDATES:
        if os.path.exists(_p):
            YOLO_WEIGHTS = _p
            break
    if YOLO_WEIGHTS is None:
        YOLO_WEIGHTS = "yolov8n.pt"  # ultralytics 会自动下载

# ---------- YOLO 模型（懒加载，只在第一次请求时加载） ----------
_yolo_model = None

def get_yolo_model():
    """懒加载 YOLO 模型，避免启动时占用显存"""
    global _yolo_model
    if not YOLO_AVAILABLE:
        return None
    if _yolo_model is None:
        if not YOLO_WEIGHTS:
            logger.error("YOLO 权重路径未配置")
            return None
        logger.info(f"正在加载 YOLO 模型: {YOLO_WEIGHTS}")
        try:
            _yolo_model = YOLO(YOLO_WEIGHTS)
            logger.info("✅ YOLO 模型加载完成")
        except Exception as e:
            logger.error(f"YOLO 模型加载失败: {e}")
            return None
    return _yolo_model


# ==================== 路由：页面 ====================

@app.route("/")
def index():
    """聊天助手主页"""
    return render_template("index.html")


@app.route("/yolo")
def yolo_page():
    """YOLO 目标检测演示页"""
    if not YOLO_AVAILABLE:
        return "YOLO 功能未就绪，请检查依赖是否安装（pip install ultralytics opencv-python）", 503
    return render_template("yolo.html")


# ==================== 路由：智能体 API 代理 ====================

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    聊天接口 —— 转发请求到 agent.py (5001)
    前端只和 5000 通信，避免跨域问题
    """
    data = request.json or {}
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not message:
        return jsonify({"error": "消息不能为空"}), 400

    try:
        resp = requests.post(
            f"{AGENT_URL}/api/chat",
            json={"message": message, "session_id": session_id},
            timeout=30
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.exceptions.ConnectionError:
        logger.error("无法连接到智能体服务")
        return jsonify({
            "reply": "❌ 智能体服务暂时不可用，请确认 agent.py 已启动且 LM Studio 已开启 Server。",
            "tools_used": []
        }), 500
    except Exception as e:
        logger.error(f"调用智能体失败: {e}")
        return jsonify({
            "reply": f"❌ 请求失败：{e}",
            "tools_used": []
        }), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    """重置会话 —— 转发到 agent.py"""
    data = request.json or {}
    session_id = data.get("session_id", "default")
    try:
        requests.post(
            f"{AGENT_URL}/api/reset",
            json={"session_id": session_id},
            timeout=5
        )
    except Exception as e:
        logger.warning(f"重置智能体会话失败（可忽略）: {e}")
    return jsonify({"status": "ok"})


@app.route("/api/health", methods=["GET"])
def health():
    """健康检查 —— 检查智能体服务 + YOLO 状态"""
    agent_online = False
    try:
        r = requests.get(f"{AGENT_URL}/api/health", timeout=3)
        r.raise_for_status()
        agent_data = r.json()
        agent_online = agent_data.get("model_online", False)
    except Exception:
        pass

    return jsonify({
        "web_app": "running",
        "agent_online": agent_online,
        "yolo_available": YOLO_AVAILABLE,
        "yolo_loaded": _yolo_model is not None,
        "yolo_weights": YOLO_WEIGHTS if YOLO_AVAILABLE else None,
        "timestamp": time.time()
    })


# ==================== 路由：YOLO 检测 API ====================

@app.route("/api/yolo/detect", methods=["POST"])
def yolo_detect():
    """
    接收前端传来的图片（base64），返回检测结果
    请求：{"image": "data:image/jpeg;base64,/9j/..."}
    响应：{"detections": [...], "image": "data:...", "count": N}
    """
    if not YOLO_AVAILABLE:
        return jsonify({
            "error": "YOLO 未安装，请运行: pip install ultralytics opencv-python numpy"
        }), 503

    model = get_yolo_model()
    if model is None:
        return jsonify({
            "error": f"YOLO 模型加载失败，请检查权重文件是否存在: {YOLO_WEIGHTS}"
        }), 503

    data = request.json or {}
    img_b64 = data.get("image", "")

    if not img_b64:
        return jsonify({"error": "未收到图片数据"}), 400

    try:
        # base64 → numpy array → OpenCV 格式
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "图片解码失败，请检查图片格式"}), 400

        # ---- YOLO 推理 ----
        # half=True: FP16 半精度，速度提升 ~30%
        # conf=0.5: 过滤低质量检测
        results = model(frame, half=True, conf=0.5, verbose=False)

        # ---- 提取检测结果 ----
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append({
                    "label": model.names[cls_id],
                    "confidence": round(conf, 3),
                    "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()]
                })

        # ---- 绘制检测框 ----
        annotated = results[0].plot()
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result_b64 = base64.b64encode(buffer).decode("utf-8")

        logger.info(f"YOLO 检测完成: {len(detections)} 个目标")

        return jsonify({
            "detections": detections,
            "image": f"data:image/jpeg;base64,{result_b64}",
            "count": len(detections)
        })

    except Exception as e:
        logger.error(f"YOLO 检测失败: {e}")
        return jsonify({"error": f"检测失败: {e}"}), 500


# ==================== 启动 ====================

if __name__ == "__main__":
    print("=" * 55)
    print("  🌐 Web 应用启动中...")
    print(f"  聊天界面:  http://localhost:5000")
    print(f"  YOLO 检测: http://localhost:5000/yolo")
    print(f"  健康检查:  http://localhost:5000/api/health")
    print(f"  智能体:    {AGENT_URL}")
    print(f"  YOLO 支持: {'✅ 已启用' if YOLO_AVAILABLE else '❌ 未安装'}")
    if YOLO_AVAILABLE:
        print(f"  YOLO 权重: {YOLO_WEIGHTS}")
    print("=" * 55)

    app.run(host="0.0.0.0", port=5000, debug=False)
