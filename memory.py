"""
memory.py —— 华小牛 Harness 的记忆模块
=====================================
负责保存任务状态、执行日志、历史记录。
面试考察点：状态持久化、日志规范。
"""

import json
import os
import time
from datetime import datetime

MEMORY_DIR = "harness_memory"
STATE_FILE = os.path.join(MEMORY_DIR, "state.json")
LOG_FILE = os.path.join(MEMORY_DIR, "execution.log")


def init_memory():
    """初始化记忆目录"""
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"tasks": [], "current_plan": [], "step_index": 0}, f, ensure_ascii=False, indent=2)


def save_state(state: dict):
    """保存当前状态到文件"""
    init_memory()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_state() -> dict:
    """加载状态"""
    init_memory()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"tasks": [], "current_plan": [], "step_index": 0}


def log_execution(step_name: str, status: str, detail: str = ""):
    """记录执行日志（追加）"""
    init_memory()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {step_name} | {status} | {detail}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)


def add_task(goal: str):
    """添加新任务到状态"""
    state = load_state()
    task = {
        "goal": goal,
        "start_time": time.time(),
        "end_time": None,
        "status": "running"
    }
    state["tasks"].append(task)
    save_state(state)
    return len(state["tasks"]) - 1  # 返回任务索引


def update_task_status(task_index: int, status: str):
    """更新任务状态"""
    state = load_state()
    if 0 <= task_index < len(state["tasks"]):
        state["tasks"][task_index]["status"] = status
        if status in ["completed", "failed"]:
            state["tasks"][task_index]["end_time"] = time.time()
        save_state(state)


def save_plan(plan: list):
    """保存当前计划"""
    state = load_state()
    state["current_plan"] = plan
    state["step_index"] = 0
    save_state(state)


def get_current_plan() -> list:
    """获取当前计划"""
    state = load_state()
    return state["current_plan"]


def get_step_index() -> int:
    """获取当前步骤索引"""
    state = load_state()
    return state["step_index"]


def advance_step():
    """推进步骤索引"""
    state = load_state()
    state["step_index"] += 1
    save_state(state)


if __name__ == "__main__":
    # 测试
    init_memory()
    print("记忆模块测试：")
    task_idx = add_task("测试任务")
    print(f"添加任务，索引：{task_idx}")
    log_execution("测试步骤", "success", "一切正常")
    print("日志已写入")