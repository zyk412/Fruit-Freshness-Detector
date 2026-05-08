import os
import sys
# 确保能找到同目录下的 spe_7341
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from spe_7341 import run_analysis

class FreshnessAnalyzer:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def analyze(self):
        """
        执行光谱分析流程
        """
        print("\n[Spectrum] 启动光谱分析内核...")
        try:
            # 直接调用你修改后的 spe_7341 运行函数
            run_analysis()
            return True
        except Exception as e:
            print(f"❌ 光谱分析发生异常: {e}")
            return False

# 调试用
if __name__ == "__main__":
    analyzer = FreshnessAnalyzer()
    analyzer.analyze()