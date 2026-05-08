import os
import sys
import time

# --- 关键：动态添加搜索路径 ---
# 获取当前 main.py 所在的根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 将算法子目录添加到 sys.path，这样 Python 就能找到它们
sys.path.append(os.path.join(BASE_DIR, "algorithm", "fruit_detection"))
sys.path.append(os.path.join(BASE_DIR, "algorithm", "spectrum_analysis"))
sys.path.append(os.path.join(BASE_DIR, "firmware"))

# --- 导入各模块 ---
try:
    from firmware.auto_capture2 import take_photo
    from algorithm.fruit_detection.preprocess import ImageProcessor
    # 假设你的 image_model.py 里封装了 FruitClassifier 类
    from algorithm.fruit_detection.image_model import ImageClassifier 
    # 导入光谱分析封装类
    from algorithm.spectrum_analysis.spectrum_7341_model import FreshnessAnalyzer
except ImportError as e:
    print(f"❌ 模块导入失败，请检查文件夹结构: {e}")
    sys.exit(1)

def run_main_pipeline():
    print("=== 果实新鲜度检测自动化链路启动 ===")
    
    # 1. 硬件拍照
    print("\n[阶段 1] 触发硬件采集...")
    if not take_photo():
        print("❌ 拍照失败，终止流程。")
        return

    # 2. 图像识别
    print("\n[阶段 2] 图像预处理与品种识别...")
    # 定位最新图片
    raw_data_dir = os.path.join(BASE_DIR, "data", "raw")
    processor = ImageProcessor()
    latest_img_path = processor.get_latest_image(raw_data_dir)
    
    if not latest_img_path:
        print("❌ 未在 data/raw 中找到新图片。")
        return

    # 预处理并识别
    processed_frame = processor.process(latest_img_path)
    # 加载模型并预测
    model_path = os.path.join(BASE_DIR, "algorithm", "fruit_detection", "yolov8n-cls.pt")
    classifier = ImageClassifier(model_path)
    fruit_name = classifier.predict(processed_frame)
    print(f"✅ 视觉识别结论: {fruit_name}")

    # 将识别结果写入光谱模块需要的 fruit.txt
    config_path = os.path.join(BASE_DIR, "algorithm", "spectrum_analysis", "fruit.txt")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(fruit_name)
    print(f"品种信息已同步至光谱模块。")

    # 3. 光谱新鲜度分析
    print("\n[阶段 3] 启动多光谱定量分析...")
    analyzer = FreshnessAnalyzer()
    success = analyzer.analyze()
    
    if success:
        print("\n=== 系统任务全部完成 ===")
    else:
        print("\n❌ 光谱分析环节出现错误。")

if __name__ == "__main__":
    run_main_pipeline()