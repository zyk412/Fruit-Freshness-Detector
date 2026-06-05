import os
import sys

print("1. 开始测试路径...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "algorithm", "fruit_detection"))
sys.path.append(os.path.join(BASE_DIR, "algorithm", "spectrum_analysis"))
sys.path.append(os.path.join(BASE_DIR, "firmware"))

print(f"当前根目录: {BASE_DIR}")

try:
    print("2. 尝试导入 ImageClassifier...")
    from algorithm.fruit_detection.image_model import ImageClassifier
    print("✅ ImageClassifier 导入成功")
    
    print("3. 尝试导入 FreshnessAnalyzer...")
    from algorithm.spectrum_analysis.spectrum_7341_model import FreshnessAnalyzer
    print("✅ FreshnessAnalyzer 导入成功")
    
    print("4. 尝试导入 take_photo...")
    from firmware.auto_capture2 import take_photo
    print("✅ take_photo 导入成功")
    
except Exception as e:
    print(f"❌ 导入出错: {e}")

print("\n--- 测试完成 ---")