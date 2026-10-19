"""
YOLO 实时推理 - Python 快速实现
===============================
参考 ultralytics 官方文档，不自研。
按 'q' 退出，按 's' 保存当前帧。
"""
import os
import cv2
from ultralytics import YOLO


def get_model_path():
    """自动查找可用的权重文件"""
    candidates = [
        r"D:\ai-project\runs\detect\exp_yolov8n\weights\best.pt",   # 自定义训练权重
        r"D:\ai-project\runs\detect\runs\detect\exp_yolov8n\weights\best.pt",  # 嵌套目录（兼容旧路径）
        r"D:\ai-project\yolov8n.pt",                                # 预训练权重
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "yolov8n.pt"  # ultralytics 会自动下载


def main():
    model_path = get_model_path()
    print(f"📦 加载模型: {model_path}")

    model = YOLO(model_path)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    # 设置分辨率（降低分辨率 = 提高 FPS）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("✅ 摄像头已打开")
    print("   按 'q' 退出 | 按 's' 保存当前帧")
    print(f"   模型: {model_path}")

    prev_tick = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取摄像头帧失败")
            break

        # ---- 推理 ----
        # half=True: FP16 半精度，速度提升 ~30%
        # conf=0.5: 置信度阈值，过滤低质量检测
        results = model(frame, half=True, conf=0.5, verbose=False)

        # ---- 绘制结果 ----
        annotated = results[0].plot()

        # ---- 计算并显示 FPS ----
        curr_tick = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_tick - prev_tick)
        prev_tick = curr_tick
        cv2.putText(annotated, f"FPS: {fps:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ---- 显示检测框数量 ----
        box_count = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated, f"Objects: {box_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("YOLO Real-time Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{int(curr_tick)}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"   📸 已保存: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 已退出")


if __name__ == "__main__":
    main()