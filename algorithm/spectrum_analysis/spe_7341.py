import pandas as pd
import numpy as np
import os


def get_fruit_config(fruit_name):
    """
    根据品种返回计算权重 (a: 颜色权重, b: 结构权重)、基准值和判断阈值
    """
    configs = {
        "Apple": {"a": 1.8, "b": 0.6, "cfi_base": 0.12, "ssi_base": 0.14, "warn": 0.82},
        "Banana": {"a": 1.5, "b": 0.8, "cfi_base": 0.10, "ssi_base": 0.15, "warn": 0.85},
        "Strawberry": {"a": 0.5, "b": 2.2, "cfi_base": 0.15, "ssi_base": 0.12, "warn": 0.88},
        "Tomato": {"a": 0.8, "b": 1.5, "cfi_base": 0.11, "ssi_base": 0.13, "warn": 0.85}
    }
    # 如果识别出的品种不在列表中，使用默认值
    return configs.get(fruit_name, {"a": 1.0, "b": 1.0, "cfi_base": 0.12, "ssi_base": 0.14, "warn": 0.85})


def calculate_score(row, a, b, cfi_base, ssi_base):
    """
    使用国标算法计算新鲜度得分
    AS7341 通道: F1(415nm), F5(555nm), F8(680nm), NIR
    """
    try:
        # 提取关键波段
        f1_415 = float(row[0])  # 蓝色/紫色
        f5_555 = float(row[4])  # 黄绿色
        f8_680 = float(row[7])  # 深红色
        nir = float(row[9])  # 近红外

        # 防止除零
        if f5_555 == 0 or f8_680 == 0:
            return 0

        # 计算子指数（国标公式）
        cfi = f1_415 / f5_555  # 颜色新鲜度指数
        ssi = nir / f8_680  # 结构稳定指数

        # 归一化（除以国标基准值）
        cfi_norm = np.clip(cfi / cfi_base, 0, 2.0)  # 限制范围避免异常值
        ssi_norm = np.clip(ssi / ssi_base, 0, 2.0)

        # 加权计算得分（0-100范围）
        score = (cfi_norm ** a) * (ssi_norm ** b) * 100

        return score

    except (IndexError, ValueError) as e:
        return 0


def run_analysis():
    # --- 1. 动态路径处理 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    txt_path = os.path.join(current_dir, 'fruit.txt')
    csv_path = os.path.join(project_root, 'firmware', 'spectral.csv')

    # --- 2. 读取品种并获取配置 ---
    if not os.path.exists(txt_path):
        print(f"❌ 错误：找不到品种文件 {txt_path}")
        return

    with open(txt_path, 'r', encoding='utf-8') as f:
        fruit_name = f.read().strip()

    config = get_fruit_config(fruit_name.capitalize())
    print(f"--- ⚖️ 正在分析品种: {fruit_name} ---")

    # --- 3. 读取光谱数据 ---
    if not os.path.exists(csv_path):
        print(f"❌ 错误：在 firmware 目录下未发现最新采集的 spectral.csv")
        print(f"请先运行 firmware/save_spectrum_oneline.py 采集数据。")
        return

    try:
        df = pd.read_csv(csv_path, header=None, engine='python')
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    # --- 4. 确定基准得分（使用第二行作为新鲜样本）---
    if len(df) < 2:
        print("❌ 错误：CSV数据行数不足（需包含校准行和新鲜样本行）")
        return

    fresh_row = df.iloc[1].values
    baseline_score = calculate_score(fresh_row, config['a'], config['b'], config['cfi_base'], config['ssi_base'])

    if baseline_score == 0:
        print("❌ 错误：无法计算基准得分，请检查 CSV 第 2 行数据")
        return

    # --- 5. 遍历并判断 ---
    for i, row in df.iterrows():
        if i == 0:
            print(f"数据 {i + 1}: 系统背景校准 - 已跳过")
            continue

        current_score = calculate_score(row.values, config['a'], config['b'], config['cfi_base'], config['ssi_base'])
        freshness_ratio = current_score / baseline_score

        # 根据配置中的 warn 阈值判断
        if freshness_ratio >= config['warn']:
            status = "新鲜 ✨"
        elif freshness_ratio >= config['warn'] * 0.8:
            status = "开始变质/不新鲜 ⚠️"
        else:
            status = "严重变质 ❌"

        print(f"数据 {i + 1}: 相对新鲜度: {freshness_ratio:.2%} -> 结论: {status}")


if __name__ == "__main__":
    run_analysis()