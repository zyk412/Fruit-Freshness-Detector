import os
from ultralytics import YOLO
import cv2

# --- 1. 路径配置 ---
model_path = r"E:\Fruit-Freshness-Detector\runs\fruit_cls_v2\weights\best.pt"
# 确保这里是你最新拍的那张香蕉图
test_img = r"E:\Fruit-Freshness-Detector\data\raw\photo_20260508_170305.jpg" 

# --- 2. 逻辑执行 ---
if not os.path.exists(model_path) or not os.path.exists(test_img):
    print("❌ 路径错误，请检查模型或图片是否存在！")
else:
    # 加载模型
    model = YOLO(model_path)
    
    # 执行推理
    results = model.predict(source=test_img, conf=0.1)
    
    # 渲染结果到图片
    res_plotted = results[0].plot()
    
    # --- 3. 弹出窗口 ---
    # 给窗口起个名字，方便识别
    win_name = "YOLO Debug - Press Any Key to Close"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) # 允许缩放窗口
    cv2.imshow(win_name, res_plotted)
    
    # 打印控制台信息方便对照
    print(f"\n🏆 判定结果: {model.names[results[0].probs.top1]}")
    print(f"📊 完整概率分布: {results[0].probs.data.tolist()}")
    
    # 关键：等待按键，否则窗口会一闪而过
    cv2.waitKey(0)
    cv2.destroyAllWindows()