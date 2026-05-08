import cv2
import numpy as np
import os
import glob

class ImageProcessor:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), median_ksize=5):
        # 初始化 CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.median_ksize = median_ksize

    def get_latest_image(self, folder_path="../../data/raw"):
        """
        自动获取文件夹下最新产生的一张图片路径
        """
        # 获取所有 .jpg 文件
        search_path = os.path.join(folder_path, "*.jpg")
        files = glob.glob(search_path)
        if not files:
            return None
        # 按修改时间排序，取最后一张
        latest_file = max(files, key=os.path.getmtime)
        return latest_file

    def process(self, image_path):
        """
        执行预处理逻辑：中值滤波 + CLAHE
        """
        # 读取图片
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ 无法读取图片: {image_path}")
            return None

        # 1. 中值滤波 (去噪)
        denoised = cv2.medianBlur(frame, self.median_ksize)

        # 2. LAB 空间下的 CLAHE (增强对比度)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        return final_img

# --- 调试用测试逻辑 ---
if __name__ == "__main__":
    processor = ImageProcessor()
    
    # 自动获取最新照片的名称
    raw_folder = "../../data/raw" # 注意这里是相对路径
    img_path = processor.get_latest_image(raw_folder)
    
    if img_path:
        print(f"正在处理最新图片: {img_path}")
        result = processor.process(img_path)
        
        # 保存或展示
        cv2.imshow("Original vs Processed", result)
        cv2.waitKey(0)
    else:
        print("文件夹内暂时没有图片。")