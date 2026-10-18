from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# LM Studio 本地 API 配置
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "qwen2.5-7b-instruct"

import requests

# 存储每个用户的对话历史（简单版，重启后清空）
# 格式：{ session_id: [ {role, content}, ... ] }
chat_histories = {}

@app.route("/")
def index():
    """首页，返回聊天界面"""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """接收前端消息，调用本地模型，返回回复"""
    data = request.json
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default")

    # 初始化该用户的对话历史
    if session_id not in chat_histories:
        chat_histories[session_id] = [
            {
                "role": "system",
                "content": "你是一个友善、有趣的AI助手。回答简洁有条理，适当使用emoji。"
            }
        ]

    # 加入用户消息
    chat_histories[session_id].append({
        "role": "user",
        "content": user_message
    })

    try:
        # 调用 LM Studio API
        response = requests.post(API_URL, json={
            "model": MODEL,
            "messages": chat_histories[session_id],
            "temperature": 0.7,
            "max_tokens": 500
        }, timeout=30)

        reply = response.json()["choices"][0]["message"]["content"]

        # 把 AI 回复也加入历史
        chat_histories[session_id].append({
            "role": "assistant",
            "content": reply
        })

        return jsonify({"reply": reply, "status": "success"})

    except Exception as e:
        return jsonify({"reply": f"出错了：{str(e)}", "status": "error"})

@app.route("/reset", methods=["POST"])
def reset():
    """重置对话"""
    session_id = request.json.get("session_id", "default")
    chat_histories.pop(session_id, None)
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("🌐 Web AI 助手已启动！浏览器打开：http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)