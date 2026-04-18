# algorithm/fruit_detection/image_model.py
from ultralytics import YOLO

class FruitImageClassifier:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_prediction(self, img_path):
        results = self.model.predict(source=img_path, verbose=False)
        result = results[0]
        
        # 返回约定的接口格式
        return {
            "label": result.names[result.probs.top1],
            "confidence": float(result.probs.top1conf.item())
        }