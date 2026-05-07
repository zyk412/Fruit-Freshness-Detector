from ultralytics import YOLO

# 1. 加载一个预训练模型 (会自动下载约 6MB 的文件)
model = YOLO('yolov8n.pt')

# 2. 对官方示例图进行推理
# save=True 会把结果保存在 runs/detect/predict 文件夹下
results = model.predict(source='https://ultralytics.com/images/bus.jpg', save=True)

# 3. 打印简要结果
print("环境测试成功！模型已识别出物体。")