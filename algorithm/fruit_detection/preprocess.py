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
        自动获取最新产生的一张图片路径（测试用）
        """
        search_path = os.path.join(folder_path, "**", "*.jpg")
        search_path = search_path.replace("\\", "/")
        files = glob.glob(search_path, recursive=True)
        if not files:
            return None
        latest_file = max(files, key=os.path.getmtime)
        return latest_file

    def process(self, image_path):
        """
        执行预处理逻辑：中值滤波 + CLAHE
        """
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

    def process_all_images(self, folder_path):
        """
        🔥🔥🔥 新增：批量处理文件夹下的所有水果照片
        """
        search_path = os.path.join(folder_path, "**", "*.jpg")
        search_path = search_path.replace("\\", "/")
        files = glob.glob(search_path, recursive=True)
        
        # 过滤掉已经处理过的图片（防止二次处理冲突）
        files = [f for f in files if "processed_" not in os.path.basename(f)]
        
        total_files = len(files)
        print(f"📦 找到待处理的原始图片共: {total_files} 张")
        
        count = 0
        for img_path in files:
            result = self.process(img_path)
            if result is not None:
                # 直接覆盖原图，或者保存为 processed_ 开头的文件
                # 为了后续 organize_data.py 和训练集不容易出错，这里直接【覆盖原图】是最省心的
                # 如果你想保留原图，可以改成：output_path = os.path.join(dir_name, "processed_" + file_name)
                cv2.imwrite(img_path, result)
                count += 1
                if count % 50 == 0 or count == total_files:
                    print(f"⏳ 进度: [{count}/{total_files}] 张图片预处理完成...")
                    
        print(f"🎉 批量预处理大功告成！成功处理 {count} 张图片。")

# --- 运行批量处理 ---
if __name__ == "__main__":
    processor = ImageProcessor()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_folder = os.path.normpath(os.path.join(current_dir, "../../data/raw"))
    
    print(f"📂 正在扫描的数据根目录: {raw_folder}")
    
    # ⬇️ 调用批量处理函数
    processor.process_all_images(raw_folder)