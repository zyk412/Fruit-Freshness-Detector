import os
import sys
import argparse
import pandas as pd
from prettytable import PrettyTable

# 1. 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'algorithm', 'fruit_detection'))
sys.path.append(os.path.join(BASE_DIR, 'algorithm', 'spectrum_analysis'))

try:
    from algorithm.fruit_detection.image_model import ImageClassifier
    from algorithm.spectrum_analysis.spectrum_model import SpectrumAnalyzer
except ImportError as e:
    print(f"模块导入失败，请检查文件夹结构：{e}")
    sys.exit(1)

def main():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="水果新鲜度智能检测流")
    parser.add_argument("img_id", nargs="?", default="1", help="测试图片的序号")
    args = parser.parse_args()

    # 路径定义
    spectral_csv = os.path.join(BASE_DIR, 'algorithm', 'spectrum_analysis', 'spectral.csv')
    txt_file = os.path.join(BASE_DIR, 'algorithm', 'spectrum_analysis', 'fruit.txt')
    image_name = f"test_img_{args.img_id}.jpg"
    test_image = os.path.join(BASE_DIR, image_name)

    if not os.path.exists(spectral_csv) or not os.path.exists(test_image):
        print("致命错误：找不到光谱文件或图片文件")
        return

    # 初始化模型
    img_model = ImageClassifier()
    spec_model = SpectrumAnalyzer()

    # --- 第一步：识别品种 ---
    raw_fruit_type = img_model.predict(test_image) 
    
    # --- 第二步：核心逻辑修复 - 检查品种是否存在 ---
    # 我们先做一次标准化处理
    fruit_key = str(raw_fruit_type).strip().capitalize()
    
    # 检查是否在 spec_model 的标准库中 (假设 FRUIT_STANDARDS 是公开属性)
    if fruit_key not in spec_model.FRUIT_STANDARDS:
        print(f"⚠️ [提示] 品种 '{fruit_key}' 不在标准库中，将采用 Apple 标准进行计算。")
        final_type = "Apple"
    else:
        final_type = fruit_key

    # --- 第三步：同步写入 fruit.txt ---
    # 写入的是最终用于计算的品种名
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"{final_type}\n")

    # --- 第四步：读取光谱并计算 ---
    df = pd.read_csv(spectral_csv)
    raw_values = df.iloc[0].values[:14]
    
    # 使用 final_type 确保计算过程和执行标准完全匹配
    score, std_id = spec_model.calculate_score(raw_values, final_type)

    # --- 第五步：表格展示 ---
    table = PrettyTable()
    table.field_names = ["序号", "实际识别", "计算标准", "新鲜度 (%)", "执行标准"]
    
    # 在表格中同时展示“实际识别”和“计算标准”，方便你核对逻辑
    table.add_row([
        args.img_id, 
        raw_fruit_type, # YOLO 原始结果
        final_type,     # 实际用于查表的品种
        f"{score:.2f}", 
        std_id
    ])

    print(table)
    print(f"检测完成。")

if __name__ == "__main__":
    main()