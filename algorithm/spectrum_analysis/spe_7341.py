import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 国标参数库
# ==========================================
FRUIT_STANDARDS = {
    'Apple': {
        'cfi_base': 0.12, 'ssi_base': 0.14,
        'weight_a': 1.8, 'weight_b': 0.6,
        'std_id': 'GB/T 10651'
    },
    'Strawberry': {
        'cfi_base': 0.15, 'ssi_base': 0.12,
        'weight_a': 0.5, 'weight_b': 2.2,
        'std_id': 'GH/T 1012'
    },
    'Banana': {
        'cfi_base': 0.10, 'ssi_base': 0.15,
        'weight_a': 1.5, 'weight_b': 0.8,
        'std_id': 'NY/T 357'
    }
}

def calculate_pure_score(raw_data, fruit_name):
    fruit_key = str(fruit_name).strip().capitalize()
    if fruit_key not in FRUIT_STANDARDS:
        fruit_key = 'Apple'

    std = FRUIT_STANDARDS[fruit_key]

    # AS7341 索引: 0=415nm, 4=555nm, 7=680nm, 9=NIR
    cfi = raw_data[0] / raw_data[4] if raw_data[4] > 0 else 0
    ssi = raw_data[9] / raw_data[7] if raw_data[7] > 0 else 0

    cfi_norm = np.clip(cfi / std['cfi_base'], 0, 1.0)
    ssi_norm = np.clip(ssi / std['ssi_base'], 0, 1.0)

    score = (cfi_norm ** std['weight_a']) * (ssi_norm ** std['weight_b']) * 100
    return round(score, 2), std['std_id']

# ==========================================
# 2. 执行流（直接使用绝对路径）
# ==========================================
def main():
    # 直接填入你给的绝对地址
    csv_file = r'D:\work\双创\cal\spectrum_data.csv'
    txt_file = r'D:\work\双创\cal\fruit.txt'

    # 读取数据
    df = pd.read_csv(csv_file)
    with open(txt_file, 'r', encoding='utf-8') as f:
        fruit_list = [line.strip() for line in f if line.strip()]

    print(f"{'编号':<6} | {'检测品种':<12} | {'量化新鲜度 (%)':<15} | {'参考标准'}")
    print("-" * 60)

    for idx, (data_row, fruit_type) in enumerate(zip(df.values, fruit_list)):
        raw_values = data_row[:10] # AS7341 取前10列
        score, std_id = calculate_pure_score(raw_values, fruit_type)
        print(f"{idx:<8} | {fruit_type:<12} | {score:<15.2f} | {std_id}")

if __name__ == "__main__":
    main()