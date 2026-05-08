import os
import sys
import time

# 强制刷新输出，防止缓冲区导致控制台不显示
sys.stdout.reconfigure(line_buffering=True)

# --- 1. 路径设置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "algorithm", "fruit_detection"))
sys.path.append(os.path.join(BASE_DIR, "algorithm", "spectrum_analysis"))
sys.path.append(os.path.join(BASE_DIR, "firmware"))

print(">>> [系统初始化] 路径映射已完成...")

# --- 2. 模块导入 ---
try:
    from firmware.auto_capture2 import take_photo
    from algorithm.fruit_detection.preprocess import ImageProcessor
    from algorithm.fruit_detection.image_model import ImageClassifier 
    from algorithm.spectrum_analysis.spectrum_7341_model import FreshnessAnalyzer
    print(">>> [系统初始化] 核心算法库加载成功!")
except Exception as e:
    print(f"❌ [初始化失败] 导入模块时出错: {e}")
    sys.exit(1)

def run_main_pipeline():
    print("\n" + "="*40)
    print("      🍎 果实新鲜度检测自动化系统")
    print("="*40)
    
    # --- 阶段 1: 硬件采集 ---
    print("\n[阶段 1] 正在连接 ESP32 并触发拍照...")
    # 如果程序卡在这里不动，请检查 ESP32 是否被其他程序占用
    try:
        start_time = time.time()
        success = take_photo() 
        if not success:
            print("❌ 拍照失败: 硬件响应超时或未连接。")
            return
        print(f"✅ 硬件采集耗时: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"💥 硬件阶段崩溃: {e}")
        return

    # --- 阶段 2: 图像识别 ---
    print("\n[阶段 2] 正在加载神经网络并识别品种...")
    try:
        raw_dir = os.path.join(BASE_DIR, "data", "raw")
        processor = ImageProcessor()
        img_path = processor.get_latest_image(raw_dir)
        
        if not img_path:
            print(f"❌ 错误: 在 {raw_dir} 下找不到刚刚拍摄的照片。")
            return

        # 动态加载模型（使用你的最新路径）
        model_path = os.path.join(BASE_DIR, "runs", "fruit_cls_v2", "weights", "best.pt")
       
        if not os.path.exists(model_path):
            print(f"❌ 严重错误：在 {model_path} 找不到权重文件！")
            print("请检查 runs 文件夹是否确实在根目录下。")
            return
       
        classifier = ImageClassifier()
        
        fruit_name = classifier.predict(img_path)
        print(f"✅ 识别结果: 【{fruit_name}】")

        # 同步品种信息
        txt_path = os.path.join(BASE_DIR, "algorithm", "spectrum_analysis", "fruit.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(fruit_name)
            f.flush()  # 强制写入文件，确保后续读取到最新品种
    except Exception as e:
        print(f"💥 视觉分析阶段崩溃: {e}")
        return

    # --- 阶段 3: 光谱分析 ---
    print("\n[阶段 3] 正在读取传感器数据并计算新鲜度...")
    try:
        analyzer = FreshnessAnalyzer()
        # 确保 spe_7341.py 里的路径已经按我刚才说的改好了
        analyzer.analyze() 
    except Exception as e:
        print(f"💥 光谱分析阶段崩溃: {e}")

    print("\n" + "="*40)
    print("      ✨ 全流程处理完毕")
    print("="*40)

if __name__ == "__main__":
    # 强制立刻运行
    run_main_pipeline()
    input("\n程序运行结束，按回车键关闭窗口...")