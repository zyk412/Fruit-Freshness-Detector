import os
from ultralytics import YOLO

class ImageClassifier:
    def __init__(self, model_path=None):
        """
        初始化分类模型
        :param model_path: 如果传入路径则使用传入路径，否则使用默认动态计算路径
        """
        if model_path is None:
            # --- 1. 动态获取默认路径 (保留你原来的逻辑) ---
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 向上退两级到 algorithm，再退一级到根目录
            project_root = os.path.dirname(os.path.dirname(current_dir))
            model_path = os.path.join(project_root, 'runs', 'fruit_cls_v2', 'weights', 'best.pt')
        
        # 2. 加载 YOLOv8 分类模型
        if not os.path.exists(model_path):
            # 如果 best.pt 还没训练好，可以尝试加载文件夹下的 yolov8n-cls.pt 作为备选
            fallback_path = os.path.join(os.path.dirname(model_path), "yolov8n-cls.pt")
            if os.path.exists(fallback_path):
                model_path = fallback_path
            else:
                raise FileNotFoundError(f"找不到权重文件: {model_path}")
            
        self.model = YOLO(model_path)
        print(f"[ImageModel] 成功加载模型: {model_path}")

    def predict(self, source):
        """
        输入图片路径或处理后的矩阵，返回识别出的水果名称
        """
        # source 可以是路径字符串，也可以是 preprocess.py 处理后的图像矩阵
        results = self.model.predict(source=source, verbose=False)
        
        if len(results) > 0 and results[0].probs is not None:
            class_id = results[0].probs.top1
            fruit_name = results[0].names[class_id]
            return fruit_name
        
        return "Unknown"