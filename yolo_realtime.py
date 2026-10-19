"""
YOLO 实时推理 - Python 快速实现
===============================
参考 ultralytics 官方文档，不自研。
按 'q' 退出，按 's' 保存当前帧。
"""
import cv2
from ultralytics import YOLO


def main():
    # 加载训练好的权重（使用绝对路径，避免目录嵌套导致找不到）
    model_path = r"D:\ai-project\runs\detect\runs\detect\exp_yolov8n\weights\best.pt"

    # 如果自定义权重不存在，回退到官方预训练权重
    if not __import__('os').path.exists(model_path):
        print(f"⚠️ 自定义权重不存在: {model_path}")
        model_path = "yolov8n.pt"
        print(f"   将使用官方预训练权重: {model_path}")

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
