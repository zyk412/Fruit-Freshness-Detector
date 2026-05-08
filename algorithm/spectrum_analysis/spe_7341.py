import pandas as pd
import numpy as np
import os

def get_fruit_config(fruit_name):
    """
    根据品种返回计算权重 (a: 颜色权重, b: 结构权重) 和判断阈值
    """
    configs = {
        "Apple":      {"a": 1.8, "b": 0.6, "warn": 0.82}, # 苹果：变质时褐变明显，a权重高
        "Banana":     {"a": 1.5, "b": 0.8, "warn": 0.85},
        "Strawberry": {"a": 0.5, "b": 2.0, "warn": 0.88}, # 草莓：变质时组织溃烂，b权重高
        "Tomato":     {"a": 0.8, "b": 1.5, "warn": 0.85}
    }
    return configs.get(fruit_name, {"a": 1.0, "b": 1.0, "warn": 0.85})

def calculate_score(row, a, b):
    """
    核心算法：根据AS7343 14通道数据计算原始得分
    """
    try:
        # 归一化参考点：第12位是 Clear 通道
        clear = float(row[12])
        if clear == 0: return 0

        # 提取关键波段
        # CFI (颜色新鲜度): 405nm(Index 0) + 425nm(Index 1) 对比 515nm(Index 4) + 555nm(Index 5)
        r_f1_f2 = (float(row[0]) + float(row[1])) / clear
        r_f5_f6 = (float(row[4]) + float(row[5])) / clear

        # SSI (结构稳定性): NIR(Index 13) 对比 690nm(Index 8)
        r_f8 = float(row[8]) / clear
        r_nir = float(row[13]) / clear

        # 计算子指数
        CFI = r_f1_f2 / (r_f5_f6 + 1e-6)
        SSI = r_nir / (r_f8 + 1e-6)

        # 加权计算总分
        return (CFI ** a) * (SSI ** b)
    except (IndexError, ValueError):
        return 0

def run_analysis():
    # 使用 r"" 避免 Windows 路径转义问题
    txt_path = r'D:\work\双创\cal\fruit.txt'
    csv_path = r'D:\work\双创\cal\spectral.csv'

    # 1. 读取品种
    if not os.path.exists(txt_path):
        print(f"错误：找不到文件 {txt_path}")
        return
    with open(txt_path, 'r', encoding='utf-8') as f:
        fruit_name = f.read().strip()

    config = get_fruit_config(fruit_name)
    print(f"--- 正在分析品种: {fruit_name} ---")

    # 2. 读取光谱CSV数据
    if not os.path.exists(csv_path):
        print(f"错误：找不到文件 {csv_path}")
        return

    # 读取CSV，确保处理可能存在的格式问题
    df = pd.read_csv(csv_path, header=None, engine='python')

    # 3. 确定基准得分 (数据2为新鲜样本，对应 DataFrame 索引 1)
    if len(df) < 2:
        print("错误：CSV数据行数不足（至少需要包含基准数据行）")
        return

    fresh_row = df.iloc[1].values
    baseline_score = calculate_score(fresh_row, config['a'], config['b'])

    if baseline_score == 0:
        print("错误：无法计算基准得分，请检查新鲜样本数据")
        return

    # 4. 遍历并判断
    for i, row in df.iterrows():
        # 数据1：白色校准板
        if i == 0:
            print(f"数据 {i+1}: 系统校准行 - 已跳过")
            continue

        current_score = calculate_score(row.values, config['a'], config['b'])
        freshness_ratio = current_score / baseline_score

        # 判断状态
        if freshness_ratio >= config['warn']:
            status = "新鲜 ✨"
        elif freshness_ratio >= config['warn'] * 0.8:
            status = "开始变质/不新鲜 ⚠️"
        else:
            status = "严重变质 ❌"

        print(f"数据 {i+1}: 相对新鲜度: {freshness_ratio:.2%} -> 结论: {status}")

if __name__ == "__main__":
    run_analysis()