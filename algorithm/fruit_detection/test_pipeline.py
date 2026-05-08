import cv2
from ultralytics import YOLO
import os
import glob
import sys

# 关键：这行代码确保脚本能找到同目录下的 preprocess.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import ImageProcessor 

def run_pipeline():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 加载模型 (使用绝对路径最稳)
    model_path = os.path.join(current_dir, "yolov8n-cls.pt")
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}")
        return
    model = YOLO(model_path)
    
    # 2. 初始化预处理
    processor = ImageProcessor()
    
    # 3. 定位图片目录 (相对于当前脚本向上跳两层)
    raw_dir = os.path.abspath(os.path.join(current_dir, "../../data/raw"))
    
    # 自动获取最新照片
    files = glob.glob(os.path.join(raw_dir, "*.jpg"))
    if not files:
        print(f"目录中没有图片: {raw_dir}")
        return
    latest_img_path = max(files, key=os.path.getmtime)
    
    print(f"正在处理并识别: {os.path.basename(latest_img_path)}")

    # 4. 预处理 -> 推理
    processed_img = processor.process(latest_img_path)
    if processed_img is not None:
        # 推理并将结果画在图上
        results = model.predict(source=processed_img, conf=0.8)
        
        # 展示结果
        res_plotted = results[0].plot()
        cv2.imshow("Detection Result (Preprocessed)", res_plotted)
        
        print(f"✅ 识别成功！按下任意键关闭窗口。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()