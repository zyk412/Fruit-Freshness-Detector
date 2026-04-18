from algorithm.fruit_detection.image_model import FruitImageClassifier
# 假设这是他写的
from algorithm.spectrum_analisis.spectral_model import SpectralAnalyzer 

def main():
    # 初始化
    img_classifier = FruitImageClassifier("./runs/fruit_cls_v2/weights/best.pt")
    
    # 1. 获取图像结果
    img_data = img_classifier.get_prediction("./test_images/apple_01.jpg")
    
    # 2. 获取光谱结果 (这里先模拟他的输出)
    # spec_data = spectral_analyzer.get_score(sensor_id=1)
    spec_data = {"freshness_score": 0.85, "moisture": 0.7} # 模拟数据

    # 3. 最终判定逻辑 (接口融合)
    final_score = (img_data["confidence"] * 0.4) + (spec_data["freshness_score"] * 0.6)
    
    print(f"--- 综合检测报告 ---")
    print(f"检测品种: {img_data['label']}")
    print(f"综合新鲜度得分: {final_score:.2f}")

if __name__ == "__main__":
    main()