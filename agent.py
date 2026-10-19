"""
agent.py —— 智能体服务层
========================
职责：意图识别（前置拦截） + 工具调用 + 本地模型调度
端口：5001
依赖：flask, flask-cors, requests, python-dotenv

启动：python agent.py
"""

import os
import re
import json
import time
import logging
import datetime
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# ---------- 加载 .env 文件 ----------
load_dotenv()  # 从 .env 读取环境变量

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------- Flask 应用 ----------
app = Flask(__name__)
CORS(app)

# ==================== 配置项 ====================

# LM Studio 配置
LM_STUDIO_URL = "http://127.0.0.1:1234"
MODEL_NAME = "qwen2.5-7b-instruct"

# 高德天气 API Key（从环境变量读取，.env 文件提供）
AMAP_API_KEY = os.getenv("AMAP_API_KEY", "")

# 会话上下文存储
SESSIONS = {}

# 上下文窗口限制
MAX_CONTEXT_TURNS = 6


# ==================== 工具函数 ====================

def tool_get_time():
    """工具：获取当前时间"""
    now = datetime.datetime.now()
    weekday_map = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return {
        "result": f"当前时间是 {now.strftime('%Y年%m月%d日')} {weekday_map[now.weekday()]} {now.strftime('%H:%M:%S')}",
        "display": "🕐 系统时间"
    }


def tool_get_weather(city="广州"):
    """
    工具：查询天气（高德 API）
    """
    if not AMAP_API_KEY:
        return {
            "result": "⚠️ 天气功能未配置：缺少 AMAP_API_KEY 环境变量",
            "display": "🌤️ 天气查询"
        }

    try:
        # 1. 城市名 → adcode
        geo_url = "https://restapi.amap.com/v3/geocode/geo"
        geo_params = {"key": AMAP_API_KEY, "address": city}
        geo_resp = requests.get(geo_url, params=geo_params, timeout=5)
        geo_data = geo_resp.json()

        if geo_data.get("status") != "1" or not geo_data.get("geocodes"):
            return {
                "result": f"❌ 未找到城市「{city}」，请检查城市名称",
                "display": "🌤️ 天气查询"
            }

        adcode = geo_data["geocodes"][0]["adcode"]

        # 2. 查询天气
        weather_url = "https://restapi.amap.com/v3/weather/weatherInfo"
        weather_params = {"key": AMAP_API_KEY, "city": adcode, "extensions": "base"}
        weather_resp = requests.get(weather_url, params=weather_params, timeout=5)
        weather_data = weather_resp.json()

        if weather_data.get("status") != "1" or not weather_data.get("lives"):
            return {
                "result": f"❌ 获取「{city}」天气失败，请稍后重试",
                "display": "🌤️ 天气查询"
            }

        live = weather_data["lives"][0]
        result = (
            f"{city}当前天气：\n"
            f"🌤️ 天气：{live.get('weather', '未知')}\n"
            f"🌡️ 温度：{live.get('temperature', '未知')}°C\n"
            f"💧 湿度：{live.get('humidity', '未知')}%\n"
            f"💨 风向：{live.get('winddirection', '未知')}风 {live.get('windpower', '未知')}级\n"
            f"🕐 更新时间：{live.get('reporttime', '未知')}"
        )

        return {
            "result": result,
            "display": f"🌤️ 天气查询（{city}）"
        }

    except requests.exceptions.Timeout:
        return {
            "result": "❌ 天气查询超时，请稍后重试",
            "display": "🌤️ 天气查询"
        }
    except Exception as e:
        logger.error(f"天气查询异常: {e}")
        return {
            "result": f"❌ 天气查询失败：{e}",
            "display": "🌤️ 天气查询"
        }


# 工具注册表
TOOLS = {
    "get_time": tool_get_time,
    "get_weather": tool_get_weather,
}


# ==================== 意图识别（前置拦截） ====================

def classify_intent(text):
    """前置拦截：关键词匹配判断意图"""
    weather_keywords = ["天气", "气温", "温度", "多少度", "热不热", "冷不冷",
                        "下雨", "晴天", "阴天", "湿度", "天气预报", "会不会下雨"]
    if any(k in text for k in weather_keywords):
        city = extract_city(text) or "广州"
        return "get_weather", {"city": city}

    time_keywords = ["时间", "几点", "日期", "今天几号", "星期几", "几月几日",
                     "现在是什么时候", "当前时间", "现在几点了"]
    if any(k in text for k in time_keywords):
        return "get_time", {}

    return None, None


def extract_city(text):
    """从文本中提取城市名"""
    known_cities = [
        "北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "重庆",
        "武汉", "西安", "苏州", "天津", "长沙", "郑州", "东莞", "佛山",
        "合肥", "青岛", "沈阳", "大连", "厦门", "福州", "南昌", "昆明",
        "哈尔滨", "长春", "太原", "石家庄", "济南", "南宁", "贵阳",
        "兰州", "乌鲁木齐", "呼和浩特", "银川", "西宁", "拉萨",
        "香港", "澳门", "台北", "高雄", "台中"
    ]

    for city in known_cities:
        if city in text:
            return city

    match = re.search(r'([\u4e00-\u9fa5]{2,4})(市|区|县)', text)
    if match:
        return match.group(1) + match.group(2)

    return None


# ==================== 模型调用 ====================

def call_model(messages, system_prompt=None):
    """调用 LM Studio 本地模型"""
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False
    }

    url = f"{LM_STUDIO_URL}/v1/chat/completions"

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        logger.error("无法连接到 LM Studio")
        return "❌ 无法连接到本地模型服务，请确认 LM Studio 已加载模型并开启 Server。"
    except Exception as e:
        logger.error(f"模型调用失败: {e}")
        return f"❌ 模型调用失败：{e}"


# ==================== 路由 ====================

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not message:
        return jsonify({"error": "消息不能为空"}), 400

    if session_id not in SESSIONS:
        SESSIONS[session_id] = []

    context = SESSIONS[session_id]
    tools_used = []

    # 意图识别（前置拦截）
    tool_name, tool_args = classify_intent(message)

    if tool_name:
        logger.info(f"[{session_id}] 意图命中: {tool_name}({tool_args})")
        tool_result = TOOLS[tool_name](**tool_args)
        tools_used.append(tool_result["display"])

        system_prompt = (
            "你是一个友好的 AI 助手。用户刚才询问了相关信息，"
            "我已经帮你查到了结果。请把下面的信息用自然、友好的语气回复给用户，"
            "不要重复工具名称，直接给出结果即可。保持简洁。"
        )
        user_prompt = f"查询结果：\n{tool_result['result']}\n\n请润色成自然语言回复。"

        reply = call_model(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt
        )

        context.append({"role": "user", "content": message})
        context.append({"role": "assistant", "content": reply})
    else:
        context.append({"role": "user", "content": message})
        recent_context = context[-MAX_CONTEXT_TURNS * 2:]

        system_prompt = (
            "你是一个友好、有帮助的 AI 助手，运行在本地（离线模式）。"
            "你可以和用户聊天、回答问题。如果用户问天气或时间，"
            "请告诉用户你可以回答这类问题。"
        )

        reply = call_model(messages=recent_context, system_prompt=system_prompt)
        context.append({"role": "assistant", "content": reply})

    if len(context) > MAX_CONTEXT_TURNS * 2:
        SESSIONS[session_id] = context[-MAX_CONTEXT_TURNS * 2:]

    logger.info(f"[{session_id}] 回复: {reply[:50]}...")

    return jsonify({
        "reply": reply,
        "tools_used": tools_used
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    data = request.json or {}
    session_id = data.get("session_id", "default")
    if session_id in SESSIONS:
        SESSIONS[session_id] = []
    logger.info(f"[{session_id}] 会话已重置")
    return jsonify({"status": "ok"})


@app.route("/api/health", methods=["GET"])
def health():
    model_online = False
    try:
        url = f"{LM_STUDIO_URL}/v1/models"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            model_online = len(data.get("data", [])) > 0
    except Exception:
        pass

    return jsonify({
        "status": "running",
        "model_online": model_online,
        "model_name": MODEL_NAME,
        "lm_studio_url": LM_STUDIO_URL,
        "active_sessions": len(SESSIONS),
        "tools": list(TOOLS.keys()),
        "amap_configured": bool(AMAP_API_KEY),
        "timestamp": time.time()
    })


# ==================== 启动 ====================

if __name__ == "__main__":
    print("=" * 55)
    print("  🤖 轻量级智能体服务启动中...")
    print(f"  模型：{MODEL_NAME}")
    print(f"  LM Studio：{LM_STUDIO_URL}")
    print(f"  天气 API：{'✅ 已配置（.env）' if AMAP_API_KEY else '⚠️ 未配置'}")
    print(f"  可用工具：{list(TOOLS.keys())}")
    print(f"  服务地址：http://localhost:5001")
    print("=" * 55)

    app.run(host="0.0.0.0", port=5001, debug=False)