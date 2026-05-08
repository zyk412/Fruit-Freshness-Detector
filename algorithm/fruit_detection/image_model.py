import os
from ultralytics import YOLO

class ImageClassifier:
    def __init__(self):  # 保持只有 self，不需要外部传参
        # 1. 动态获取路径：定位到项目根目录下的 weights 文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 向上退两级到 algorithm，再退一级到根目录
        # 这里建议用 .parent 的逻辑更直观
        project_root = os.path.dirname(os.path.dirname(current_dir))
        model_path = os.path.join(project_root, 'runs', 'fruit_cls_v2', 'weights', 'best.pt')
        
        # 2. 加载 YOLOv8 分类模型
        if not os.path.exists(model_path):
            # 防御性：如果动态路径错了，尝试根目录下的直接路径
            model_path = os.path.join(os.getcwd(), 'runs', 'fruit_cls_v2', 'weights', 'best.pt')
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到权重文件: {model_path}")
            
        self.model = YOLO(model_path)
        print(f"[ImageModel] 成功加载自定义权重: {model_path}")

    def predict(self, image_path):
        """
        输入图片路径，返回识别出的水果名称（字符串）
        """
        if not os.path.exists(image_path):
            print(f"[ImageModel] 错误：图片不存在 {image_path}")
            return "Unknown"

        # 执行推理
        results = self.model.predict(source=image_path, verbose=False)
        
        # 获取索引和名称
        class_id = results[0].probs.top1
        # 关键补丁：强制转换成首字母大写，确保能匹配 spe_7341.py 里的字典
        fruit_name = results[0].names[class_id].strip().capitalize()
        
        print(f">>> 视觉识别结果: {fruit_name} (置信度: {results[0].probs.top1conf:.2%})")
        return fruit_name