"""
harness.py —— 华小牛 (HuaXiaoNiu) Harness 主程序
================================================
参考 pi-agent 架构，实现自动化视觉实验闭环。
修复：load_state 导入、NoneType 防护、JSON 解析容错。

运行方式：
    python harness.py
    （然后输入你的目标，如："自动完成YOLOv8n的训练并生成日志"）
"""

import sys
import os
import time
import json

# 将当前目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 从 memory 模块导入所有需要的函数（关键修复：明确导入 load_state）
from memory import (
    init_memory,
    add_task,
    update_task_status,
    save_plan,
    get_current_plan,
    get_step_index,
    advance_step,
    log_execution,
    load_state,          # ← 修复 NameError
)

# 从 tools_registry 导入
from tools_registry import execute_tool, call_llm, TOOLS


class HuaXiaoNiuHarness:
    def __init__(self):
        self.name = "华小牛"
        self.max_steps = 10  # 最大步骤数，防止死循环
        init_memory()
        print(f"🐮 {self.name} Harness 初始化完成")

    def planner(self, goal: str) -> list:
        """
        规划器：调用 LLM 将目标拆解为具体步骤
        """
        print(f"\n📋 [{self.name}] 正在规划任务: {goal}")
        system_prompt = """你是一个任务规划专家。用户会给出一个目标，你需要将其拆解为一系列具体的步骤。
每个步骤应该对应一个可用的工具（tool）。可用工具列表如下：
- check_env: 检查环境（GPU、依赖、权重文件）
- run_yolo_train: 运行 YOLO 训练（可指定 data_yaml, epochs, batch, imgsz）
- run_yolo_detect: 运行 YOLO 检测（可指定 source, conf）
- write_log: 写入日志（content, filename）
- call_llm: 调用大模型（prompt, system）

请以 JSON 格式输出步骤列表，每个步骤包含：
{
  "steps": [
    {"step": 1, "action": "check_env", "args": {}, "description": "检查环境"},
    {"step": 2, "action": "run_yolo_train", "args": {"epochs": 3}, "description": "训练模型"},
    ...
  ]
}
只输出 JSON，不要有其他文字。如果不确定，至少包含 check_env 和 write_log 两步。"""

        user_prompt = f"目标：{goal}"
        response = call_llm(user_prompt, system=system_prompt)

        if response.get("status") != "success":
            print(f"❌ LLM 调用失败: {response.get('error')}")
            # 返回默认计划
            return self._default_plan(goal)

        response_text = response.get("response", "")

        # 解析 JSON（增加容错）
        try:
            # 尝试提取 JSON
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()

            plan = json.loads(json_str)
            steps = plan.get("steps", [])
            if not steps:
                raise ValueError("steps 为空")
            print(f"✅ 规划完成，共 {len(steps)} 步:")
            for s in steps:
                print(f"   {s['step']}. {s['action']} - {s.get('description', '')}")
            return steps
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ 规划 JSON 解析失败: {e}")
            print(f"   LLM 返回: {response_text[:200]}...")
            return self._default_plan(goal)

    def _default_plan(self, goal: str) -> list:
        """默认计划（当 LLM 规划失败时使用）"""
        print("   使用默认计划...")
        return [
            {"step": 1, "action": "check_env", "args": {}, "description": "检查环境"},
            {"step": 2, "action": "run_yolo_train", "args": {"epochs": 3}, "description": "训练模型（3轮快速测试）"},
            {"step": 3, "action": "write_log", "args": {"content": f"目标: {goal}\n状态: 已完成训练和检测"}, "description": "写入日志"},
        ]

    def execute_step(self, step: dict) -> bool:
        """
        执行单个步骤
        返回 True 表示成功，False 表示失败
        """
        action = step.get("action")
        args = step.get("args", {})
        description = step.get("description", action)

        print(f"\n🔧 执行步骤 {step.get('step')}: {description}")
        print(f"   工具: {action}, 参数: {args}")

        start_time = time.time()
        result = execute_tool(action, args)  # ← 始终返回字典，不会是 None
        elapsed = time.time() - start_time

        # 防御性判断：确保 result 是字典
        if not isinstance(result, dict):
            print(f"❌ 工具返回异常类型: {type(result)}")
            log_execution(action, "failed", f"返回类型异常: {type(result)}")
            return False

        if result.get("status") == "success":
            print(f"✅ 步骤完成 ({elapsed:.1f}s)")
            log_execution(action, "success", f"耗时{elapsed:.1f}s")
            return True
        else:
            error_msg = result.get("error", "未知错误")
            print(f"❌ 步骤失败: {error_msg}")
            log_execution(action, "failed", error_msg)
            return False

    def run(self, human_goal: str):
        """
        主运行循环：规划 -> 逐步执行 -> 完成
        """
        print(f"\n{'='*60}")
        print(f"🚀 {self.name} 启动！")
        print(f"🎯 人类目标: {human_goal}")
        print(f"{'='*60}")

        # 1. 添加任务
        task_idx = add_task(human_goal)

        # 2. 规划
        plan = self.planner(human_goal)
        if not plan:
            print("❌ 无法生成计划，退出")
            update_task_status(task_idx, "failed")
            return

        save_plan(plan)

        # 3. 执行循环
        print(f"\n🔄 开始执行计划 (最多 {self.max_steps} 步)...")
        step_index = get_step_index()

        while step_index < len(plan) and step_index < self.max_steps:
            current_step = plan[step_index]
            print(f"\n--- 步骤 {step_index + 1}/{len(plan)} ---")

            success = self.execute_step(current_step)
            if not success:
                print("   ⚠️ 步骤失败，继续执行下一步...")

            advance_step()
            step_index = get_step_index()
            time.sleep(1)

        # 4. 完成
        if step_index >= len(plan):
            print(f"\n🎉 所有步骤执行完毕！")
            update_task_status(task_idx, "completed")
            log_execution("harness", "completed", f"目标: {human_goal}")
        else:
            print(f"\n⏹️ 达到最大步骤限制，提前终止")
            update_task_status(task_idx, "partial")

        # 5. 打印最终状态
        print(f"\n📊 最终状态:")
        try:
            state = load_state()  # ← 现在正确导入了
            print(json.dumps(state, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"   状态读取失败: {e}")


def main():
    """主入口"""
    print("🐮 欢迎使用华小牛 (HuaXiaoNiu) Harness！")
    print("提示：请确保 LM Studio 已启动并加载模型（端口 1234）")
    print("      Flask 服务 (app.py) 已启动（端口 5000）")
    print()

    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])
    else:
        goal = input("请输入你的目标（如：自动完成YOLOv8n的训练并生成日志）: ").strip()
        if not goal:
            goal = "自动完成YOLOv8n的训练并生成日志"

    harness = HuaXiaoNiuHarness()
    harness.run(goal)


if __name__ == "__main__":
    main()