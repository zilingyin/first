"""
tools_registry.py —— 华小牛 Harness 的工具库
===========================================
封装所有可调用的工具函数，包括 YOLO 训练、检测、文件操作等。
修复：Windows GBK 解码报错、返回值 None 防护。
"""

import subprocess
import os
import sys
import requests
import time
import json
import inspect

# 工具函数集合
TOOLS = {}


def register_tool(name: str):
    """装饰器：注册工具"""
    def decorator(func):
        TOOLS[name] = func
        return func
    return decorator


@register_tool("check_env")
def check_env():
    """
    检查环境：GPU、依赖、模型文件
    """
    print("  [Tool] 检查环境...")
    try:
        import torch
        env_info = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        }
        # 检查权重文件
        weights_candidates = [
            r"D:\ai-project\runs\detect\exp_yolov8n\weights\best.pt",
            r"D:\ai-project\runs\detect\runs\detect\exp_yolov8n\weights\best.pt",
            r"D:\ai-project\yolov8n.pt",
        ]
        env_info["weights_exist"] = any(os.path.exists(p) for p in weights_candidates)
        env_info["weights_path"] = next((p for p in weights_candidates if os.path.exists(p)), "Not found")
        return {"status": "success", "env": env_info}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@register_tool("run_yolo_train")
def run_yolo_train(data_yaml: str = "coco128.yaml", epochs: int = 3, batch: int = 8, imgsz: int = 640):
    """
    调用 train_yolo.py 进行训练（支持参数传递）
    默认 epochs=3 用于快速测试，正式训练可传 50
    """
    print(f"  [Tool] 启动 YOLO 训练: data={data_yaml}, epochs={epochs}, batch={batch}, imgsz={imgsz}")
    try:
        cmd = [
            sys.executable, "train_yolo.py",
            "--data", data_yaml,
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--imgsz", str(imgsz),
            "--device", "0"
        ]
        # 关键修复：强制 UTF-8 解码，忽略无法解码的字符，防止 Windows GBK 报错
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=7200  # 2小时超时
        )

        if result.returncode == 0:
            print(f"  [Tool] 训练完成")
            # 从输出中提取最后 500 字符作为结果
            output_tail = result.stdout[-500:] if result.stdout else ""
            return {"status": "success", "output_tail": output_tail}
        else:
            error_msg = result.stderr[-300:] if result.stderr else "未知错误"
            print(f"  [Tool] 训练失败: {error_msg}")
            return {"status": "failed", "error": error_msg}
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "训练超时（2小时）"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@register_tool("run_yolo_detect")
def run_yolo_detect(source: str = "0", conf: float = 0.5):
    """
    调用 YOLO 检测（通过 Flask API）
    """
    print(f"  [Tool] 启动 YOLO 检测: source={source}, conf={conf}")
    test_image = "test.jpg"
    if not os.path.exists(test_image):
        return {"status": "failed", "error": "测试图片 test.jpg 不存在，请先准备一张测试图片"}
    try:
        import base64
        with open(test_image, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        resp = requests.post(
            "http://localhost:5000/api/yolo/detect",
            json={"image": f"data:image/jpeg;base64,{img_b64}"},
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            return {"status": "success", "detections": data.get("count", 0)}
        else:
            return {"status": "failed", "error": f"API 返回 {resp.status_code}"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@register_tool("write_log")
def write_log(content: str, filename: str = "LOG.md"):
    """
    写入日志文件
    """
    print(f"  [Tool] 写入日志到 {filename}")
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"\n## Harness 执行记录 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(content)
        return {"status": "success"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@register_tool("call_llm")
def call_llm(prompt: str, system: str = "你是一个有用的AI助手。"):
    """
    调用本地大模型（LM Studio）进行推理
    """
    print(f"  [Tool] 调用 LLM，prompt 长度: {len(prompt)}")
    try:
        resp = requests.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "model": "qwen2.5-7b-instruct",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            },
            timeout=60
        )
        if resp.status_code == 200:
            data = resp.json()
            return {"status": "success", "response": data["choices"][0]["message"]["content"]}
        else:
            return {"status": "failed", "error": f"LLM 调用失败: HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def execute_tool(tool_name: str, args: dict = None) -> dict:
    """
    执行指定工具，始终返回字典
    """
    if tool_name not in TOOLS:
        return {"status": "failed", "error": f"未知工具: {tool_name}"}
    try:
        if args is None:
            args = {}
        # 根据工具函数签名传入参数
        sig = inspect.signature(TOOLS[tool_name])
        valid_args = {k: v for k, v in args.items() if k in sig.parameters}
        result = TOOLS[tool_name](**valid_args)
        # 防御：确保返回值是字典
        if not isinstance(result, dict):
            return {"status": "success", "result": str(result)}
        return result
    except Exception as e:
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    print("可用工具:", list(TOOLS.keys()))
    print("测试 check_env:")
    result = execute_tool("check_env")
    print(json.dumps(result, ensure_ascii=False, indent=2))