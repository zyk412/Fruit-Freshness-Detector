from ultralytics import YOLO
import os

def main():
    # 1. 精准定位项目根目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) 
    algorithm_dir = os.path.dirname(current_script_dir)             
    root_path = os.path.dirname(algorithm_dir)                      
    
    print(f"✅ 自动识别项目根目录: {root_path}")

    # 2. 初始化分类模型
    model = YOLO("yolov8n-cls.pt") 

    # 3. 拼接数据集和运行结果路径
    data_path = os.path.join(root_path, "algorithm", "datasets", "fruit_data", "images")
    project_path = os.path.join(root_path, "runs")

    # 提前做个双保险检查
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ 找不到数据集路径: {data_path}，请确保 organize_data.py 已经运行成功！")

    # 4. 开始训练
    print("🏋️ 混合数据集超级训练流水线启动...")
    model.train(
        data=data_path,
        epochs=50,             
        imgsz=224,             
        batch=16,              
        project=project_path, 
        name="fruit_cls_final_v2",   # 🌟 【修改点】更改为最终发布版本命名
        
        # 针对 900张实拍 + 网络图 混合特训的数据增强配置
        degrees=15.0,          
        fliplr=0.5,            
        scale=0.5,             
    )
    print(f"🏆 训练大功告成！最终生产环境权重保存在: {project_path}/fruit_cls_final_v2/weights/best.pt")

if __name__ == "__main__":
    main()