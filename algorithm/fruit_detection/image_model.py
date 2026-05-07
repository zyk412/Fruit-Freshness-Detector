import os
from ultralytics import YOLO

class ImageClassifier:
    def __init__(self):
        # 1. 动态获取路径：定位到项目根目录下的 weights 文件
        # __file__ 是当前文件路径，abspath().parent.parent.parent 回退三级到根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 根据你的截图，路径应该是：项目根目录/runs/fruit_cls_v2/weights/best.pt
        # 我们向上退两级到 algorithm，再退一级到根目录
        project_root = os.path.dirname(os.path.dirname(current_dir))
        model_path = os.path.join(project_root, 'runs', 'fruit_cls_v2', 'weights', 'best.pt')
        
        # 2. 加载 YOLOv8 分类模型
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
        # verbose=False 可以关闭推理时的详细日志，保持终端整洁
        results = self.model.predict(source=image_path, verbose=False)
        
        # 获取概率最高的类别名称
        # results[0].names 是类别字典，probs.top1 是最高分对应的索引
        class_id = results[0].probs.top1
        fruit_name = results[0].names[class_id]
        
        return fruit_name