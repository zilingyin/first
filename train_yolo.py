"""
YOLO 训练脚本
============
选择 yolov8n 预训练权重，在 COCO128 上微调 50 轮。
参数选择理由见下方注释。
"""
from ultralytics import YOLO
import torch

def main():
    # ---- 模型选择 ----
    # yolov8n: 最小最快，适合 8GB 显存实时推理
    # 如果追求精度可选 yolov8s（但推理慢 ~2x）
    model = YOLO("yolov8n.pt")  # 首次运行自动下载

    # ---- 训练 ----
    results = model.train(
        data="coco128.yaml",    # COCO128 内置数据集（128张图，80类）
        # data="dataset/data.yaml",  # 自建数据集用这行

        epochs=50,              # 50轮：COCO128 小数据集，50轮足够收敛
                                # 再多会过拟合（128张图翻来覆去）

        imgsz=640,              # 输入分辨率：640 是 YOLO 标准尺寸
                                # 小了丢细节，大了显存爆炸

        batch=8,                # 批次大小：RTX 4070 8GB 实测 batch=8 刚好
                                # 如果 OOM 就降到 4

        device=0,               # GPU 0（你的 RTX 4070）

        workers=2,              # 数据加载线程：Windows 下设 2 避免卡死
                                # Linux 可以设 4-8

        project="runs/detect",  # 输出目录
        name="exp_yolov8n",     # 实验名
        exist_ok=True,          # 允许覆盖（方便调试）

        patience=10,            # 早停：10轮无提升自动停
                                # 防止浪费时间

        save_period=10,         # 每10轮保存一次权重
                                # 方便对比不同 epoch 的效果

        verbose=True,           # 打印详细日志
    )

    # ---- 验证 ----
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")

    print("\n✅ 训练完成！")
    print(f"   最佳权重: runs/detect/exp_yolov8n/weights/best.pt")
    print(f"   最后权重: runs/detect/exp_yolov8n/weights/last.pt")

if __name__ == "__main__":
    main()