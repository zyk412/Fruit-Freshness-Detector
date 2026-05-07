from ultralytics import YOLO
import os

def main():
    # 1. 精准定位项目根目录
    # __file__ 是: .../algorithm/fruit_detection/train_cls.py
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) # .../algorithm/fruit_detection/
    algorithm_dir = os.path.dirname(current_script_dir)             # .../algorithm/
    root_path = os.path.dirname(algorithm_dir)                      # .../Fruit-Freshness-Detector/
    
    print(f"✅ 自动识别项目根目录: {root_path}")

    # 2. 初始化分类模型
    model = YOLO("yolov8n-cls.pt") 

    # 3. 拼接数据集和运行结果路径
    data_path = os.path.join(root_path, "algorithm", "datasets", "fruit_data", "images")
    project_path = os.path.join(root_path, "runs")

    # 4. 开始训练
    model.train(
        data=data_path,
        epochs=50,
        imgsz=224,
        batch=16,
        project=project_path, 
        name="fruit_cls_v2" 
    )

if __name__ == "__main__":
    main()