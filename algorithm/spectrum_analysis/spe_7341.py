import pandas as pd
import numpy as np
import os

def get_fruit_config(fruit_name):
    """
    根据品种返回计算权重 (a: 颜色权重, b: 结构权重) 和判断阈值
    """
    configs = {
        "Apple":      {"a": 1.8, "b": 0.6, "warn": 0.82},
        "Banana":     {"a": 1.5, "b": 0.8, "warn": 0.85},
        "Strawberry": {"a": 0.5, "b": 2.0, "warn": 0.88},
        "Tomato":     {"a": 0.8, "b": 1.5, "warn": 0.85}
    }
    return configs.get(fruit_name, {"a": 1.0, "b": 1.0, "warn": 0.85})

def calculate_score(row, a, b):
    """
    核心算法：根据AS7343 14通道数据计算原始得分
    """
    try:
        clear = float(row[12])
        if clear == 0: return 0

        r_f1_f2 = (float(row[0]) + float(row[1])) / clear
        r_f5_f6 = (float(row[4]) + float(row[5])) / clear
        r_f8 = float(row[8]) / clear
        r_nir = float(row[13]) / clear

        CFI = r_f1_f2 / (r_f5_f6 + 1e-6)
        SSI = r_nir / (r_f8 + 1e-6)

        return (CFI ** a) * (SSI ** b)
    except (IndexError, ValueError):
        return 0

def run_analysis():
    # --- 动态路径处理开始 ---
    # 获取当前脚本 (spe_7341.py) 所在的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 动态拼接文件名，确保在同一个目录下寻找文件
    txt_path = os.path.join(current_dir, 'fruit.txt')
    csv_path = os.path.join(current_dir, 'spectral.csv')
    # --- 动态路径处理结束 ---

    # 1. 读取品种
    if not os.path.exists(txt_path):
        print(f"❌ 错误：在脚本同级目录下找不到 {os.path.basename(txt_path)}")
        return
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        fruit_name = f.read().strip()

    config = get_fruit_config(fruit_name)
    print(f"--- 正在分析品种: {fruit_name} ---")

    # 2. 读取光谱CSV数据
    if not os.path.exists(csv_path):
        print(f"❌ 错误：在脚本同级目录下找不到 {os.path.basename(csv_path)}")
        return

    df = pd.read_csv(csv_path, header=None, engine='python')

    # 3. 确定基准得分
    if len(df) < 2:
        print("❌ 错误：CSV数据行数不足")
        return

    fresh_row = df.iloc[1].values
    baseline_score = calculate_score(fresh_row, config['a'], config['b'])

    if baseline_score == 0:
        print("❌ 错误：无法计算基准得分")
        return

    # 4. 遍历并判断
    for i, row in df.iterrows():
        if i == 0:
            print(f"数据 {i+1}: 系统校准行 - 已跳过")
            continue

        current_score = calculate_score(row.values, config['a'], config['b'])
        freshness_ratio = current_score / baseline_score

        if freshness_ratio >= config['warn']:
            status = "新鲜 ✨"
        elif freshness_ratio >= config['warn'] * 0.8:
            status = "开始变质/不新鲜 ⚠️"
        else:
            status = "严重变质 ❌"

        print(f"数据 {i+1}: 相对新鲜度: {freshness_ratio:.2%} -> 结论: {status}")

if __name__ == "__main__":
    run_analysis()