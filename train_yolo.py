"""
train_yolo.py —— YOLOv8 训练脚本
=================================
数据集：COCO128（80 类通用目标）
硬件：RTX 4070 (8GB)
参考：https://docs.ultralytics.com/modes/train/
"""

from ultralytics import YOLO
import torch

def main():
    print("=" * 50)
    print("  🚀 YOLOv8 训练开始")
    print("=" * 50)

    # 检查 GPU
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("  ⚠️ 未检测到 GPU，将使用 CPU 训练（会很慢）")

    # 加载预训练模型
    model = YOLO("yolov8n.pt")

    # 开始训练
    results = model.train(
        data="coco128.yaml",     # 数据集配置（ultralytics 会自动下载）
        epochs=50,               # 训练轮数（小数据集防过拟合）
        batch=8,                 # 批次大小（RTX 4070 实测上限）
        imgsz=640,               # 输入尺寸
        device="0" if torch.cuda.is_available() else "cpu",
        workers=4,               # 数据加载线程数
        project="runs/detect",   # 输出项目目录
        name="exp_yolov8n",      # 实验名称（输出到 runs/detect/exp_yolov8n/）
        exist_ok=True,           # 允许覆盖同名目录
        pretrained=True,         # 使用预训练权重
        optimizer="auto",        # 自动选择优化器
        verbose=True,            # 打印详细日志
        patience=10,             # 早停：10 轮无提升则停止
        save=True,               # 保存检查点
        save_period=-1,          # 不按周期保存（只在最后保存）
        cache=False,             # 不缓存数据集（节省内存）
        amp=True,                # 自动混合精度（FP16，提速省显存）
        lr0=0.01,               # 初始学习率
        lrf=0.01,               # 最终学习率（lr0 * lrf）
        momentum=0.937,         # SGD 动量
        weight_decay=0.0005,    # 权重衰减
        warmup_epochs=3.0,      # 热身轮数
        warmup_momentum=0.8,    # 热身动量
        warmup_bias_lr=0.1,     # 热身偏置学习率
        box=7.5,                # 边界框损失权重
        cls=0.5,                # 分类损失权重
        dfl=1.5,                # DFL 损失权重
    )

    print("=" * 50)
    print("  ✅ 训练完成！")
    print(f"  权重路径: runs/detect/exp_yolov8n/weights/best.pt")
    print(f"  训练曲线: runs/detect/exp_yolov8n/results.png")
    print("=" * 50)

    # 验证模型
    print("\n📊 开始验证...")
    metrics = model.val()
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
