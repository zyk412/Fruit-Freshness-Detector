from ultralytics import YOLO
import os

def main():
    # 1. 动态获取路径：向上跳两级到达项目根目录
    # __file__ 是: .../algorithm/fruit_detection/test_me.py
    current_dir = os.path.dirname(os.path.abspath(__file__)) # .../fruit_detection/
    root_path = os.path.dirname(os.path.dirname(current_dir)) # .../Fruit-Freshness-Detector/

    # 2. 拼接模型路径
    # 假设你的模型在项目根目录下的 runs 文件夹里
    model_path = os.path.join(root_path, "runs", "fruit_cls_v2", "weights", "best.pt")
    
    # 3. 拼接待测试图片的路径
    # 建议你在根目录下创建一个 test_images 文件夹放测试图，或者直接指向之前的 pict
    img_path = os.path.join(root_path, "algorithm", "datasets", "fruit_data", "images", "val", "apple", "apple_1.jpg")

    # 验证路径是否存在
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件，请检查路径: {model_path}")
        return
    
    # 4. 加载并运行推理
    print(f"✅ 正在加载模型: {model_path}")
    model = YOLO(model_path)
    
    results = model.predict(source=img_path, save=True)

    # 5. 输出结果
    for result in results:
        top1_idx = result.probs.top1
        conf = result.probs.top1conf.item()
        label = result.names[top1_idx]
        
        print("\n" + "="*30)
        print(f"识别结果: {label}")
        print(f"置信度: {conf:.2%}")
        print("="*30)

if __name__ == "__main__":
    main()