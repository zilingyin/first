"""
YOLO 实时推理 - Python 快速实现
===============================
参考 ultralytics 官方文档，不自研。
按 'q' 退出。
"""
import cv2
from ultralytics import YOLO

def main():
    # 加载训练好的权重
    model_path = "runs/detect/exp_yolov8n/weights/best.pt"
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- 推理 ----
        # half=True: FP16 半精度，速度提升 ~30%
        # conf=0.5: 置信度阈值，过滤低质量检测
        results = model(frame, half=True, conf=0.5, verbose=False)

        # ---- 绘制结果 ----
        annotated = results[0].plot()

        # ---- 显示 FPS ----
        fps_text = f"FPS: {cv2.getTickFrequency() / (cv2.getTickCount() + 1e-5):.0f}"
        cv2.putText(annotated, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Real-time Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"capture_{cv2.getTickCount()}.jpg", annotated)
            print("   📸 已保存当前帧")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 已退出")

if __name__ == "__main__":
    main()