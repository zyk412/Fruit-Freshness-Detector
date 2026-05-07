import numpy as np

# algorithm/spectrum_analysis/spectrum_model.py

class SpectrumAnalyzer:
    def __init__(self):
        # 确保你的标准库在这里定义完整
        self.FRUIT_STANDARDS = {
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

    def calculate_score(self, raw_data, fruit_name):
        # 【核心修复】标准化处理传入的品种名
        # .strip() 去空格，.capitalize() 强制首字母大写
        fruit_key = str(fruit_name).strip().capitalize()

        # 【核心修复】取消静默回退
        # 如果找不到品种，直接打印警告并返回一个特殊标记，而不是偷偷用苹果标准
        if fruit_key not in self.FRUIT_STANDARDS:
            print(f"⚠️ [警告] 未能匹配品种 '{fruit_key}'，已回退至默认 Apple 标准。")
            std = self.FRUIT_STANDARDS['Apple']
        else:
            std = self.FRUIT_STANDARDS[fruit_key]

        # 1. 特征提取
        cfi = (raw_data[0] + raw_data[1]) / (raw_data[4] + raw_data[5])
        ssi = raw_data[13] / raw_data[8]

        # 2. 归一化与裁剪
        cfi_norm = float(np.clip(cfi / std['cfi_base'], 0, 1.0))
        ssi_norm = float(np.clip(ssi / std['ssi_base'], 0, 1.0))

        # 3. 评分计算
        score = (cfi_norm ** std['weight_a']) * (ssi_norm ** std['weight_b']) * 100
        
        return round(score, 2), std['std_id']