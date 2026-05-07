import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 国标参数库 (GB/T & NY/T)
# ==========================================
# 包含各品种的国标基准值 (base) 与 生理特性权重 (weight)
FRUIT_STANDARDS = {
    'Apple': {
        'cfi_base': 1.15, 'ssi_base': 0.85,
        'weight_a': 1.8, 'weight_b': 0.6,
        'std_id': 'GB/T 10651'
    },
    'Strawberry': {
        'cfi_base': 0.90, 'ssi_base': 1.10,
        'weight_a': 0.5, 'weight_b': 2.2,
        'std_id': 'NY/T 1789'
    },
    'Pear': {
        'cfi_base': 1.05, 'ssi_base': 0.95,
        'weight_a': 1.2, 'weight_b': 1.0,
        'std_id': 'GB/T 10650'
    }
}


def calculate_pure_score(raw_data, fruit_name):
    """
    核心算法：计算绝对新鲜度得分
    """
    fruit_key = str(fruit_name).strip().capitalize()

    if fruit_key not in FRUIT_STANDARDS:
        raise ValueError(f"检测到未定义的品种: {fruit_key}")

    std = FRUIT_STANDARDS(fruit_key)

    # 1. 特征指纹提取 (比值法)
    cfi = (raw_data[0] + raw_data[1]) / (raw_data[4] + raw_data[5])
    ssi = raw_data[13] / raw_data[8]

    # 2. 国标溯源校准 (Clip 限制防止 100% 以上的逻辑溢出)
    cfi_norm = np.clip(cfi / std['cfi_base'], 0, 1.0)
    ssi_norm = np.clip(ssi / std['ssi_base'], 0, 1.0)

    # 3. 非线性加权融合评分
    # Score = (CFI^a) * (SSI^b) * 100
    score = (cfi_norm ** std['weight_a']) * (ssi_norm ** std['weight_b']) * 100
    return round(score, 2), std['std_id']


# ==========================================
# 2. 同步数据流执行
# ==========================================
# 1. 获取当前脚本 (main.py) 的绝对路径
# 结果类似: E:\Fruit-Freshness-Detector\main.py
current_script_path = os.path.abspath(__file__)

# 2. 获取项目根目录 (Fruit-Freshness-Detector)
# 结果类似: E:\Fruit-Freshness-Detector
BASE_DIR = os.path.dirname(current_script_path)

def main():
    csv_file = os.path.join(BASE_DIR, 'algorithm', 'spectrum_analysis', 'spectral.csv')
    txt_file = os.path.join(BASE_DIR, 'algorithm', 'spectrum_analysis', 'fruit.txt')

    if not os.path.exists(csv_file) or not os.path.exists(txt_file):
        print("错误：请检查 光谱数据.csv 和 fruit.txt 是否在当前文件夹")
        return

    # 读取光谱 ADC 原始数据
    df = pd.read_csv(csv_file)

    # 读取品种列表
    with open(txt_file, 'r', encoding='utf-8') as f:
        fruit_list = [line.strip() for line in f if line.strip()]

    print(f"{'编号':<6} | {'检测品种':<12} | {'量化新鲜度 (%)':<15} | {'参考标准'}")
    print("-" * 55)

    # 同步迭代：确保光谱行与品种行一一对应
    for idx, (data_row, fruit_type) in enumerate(zip(df.values, fruit_list)):
        # 提取 AS7343 的 14 个物理通道
        raw_values = data_row[:14]

        # 计算纯数值结果
        score, std_id = calculate_pure_score(raw_values, fruit_type)

        print(f"{idx:<8} | {fruit_type:<12} | {score:<15.2f} | {std_id}")

    # 行数不匹配预警
    if len(df) != len(fruit_list):
        print(f"\n[注意] 检测到数据行数不一致：光谱数据 {len(df)} 行，品种列表 {len(fruit_list)} 行。")


if __name__ == "__main__":
    main()