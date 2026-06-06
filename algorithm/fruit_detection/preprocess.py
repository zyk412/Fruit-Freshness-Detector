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
        自动获取最新产生的一张图片路径（下午真机联调/测试用）
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
        执行预处理底层核心逻辑：中值滤波 + LAB空间下的CLAHE
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

    def process_all_images(self, src_folder, dest_folder):
        """
        🌟 优雅重构版：批量处理源文件夹下的所有水果照片，并安全物理隔离输出到目标文件夹
        """
        # 规范化路径，避免操作系统斜杠差异
        src_folder = os.path.abspath(src_folder)
        dest_folder = os.path.abspath(dest_folder)

        search_path = os.path.join(src_folder, "**", "*.jpg")
        search_path = search_path.replace("\\", "/")
        files = glob.glob(search_path, recursive=True)
        
        total_files = len(files)
        if total_files == 0:
            print(f"⚠️ 在源目录中没有找到任何 .jpg 原始图片: {src_folder}")
            return

        print(f"📦 找到待处理的原始图片共: {total_files} 张")
        print(f"📥 原始图片输入目录: {src_folder}")
        print(f"📤 预处理图像输出目录: {dest_folder}")
        
        count = 0
        for img_path in files:
            img_path = os.path.abspath(img_path)
            result = self.process(img_path)
            
            if result is not None:
                # 🌟 计算相对路径，以便完美保持原始文件夹中的子目录结构（如果有的话）
                rel_path = os.path.relpath(img_path, src_folder)
                output_path = os.path.join(dest_folder, rel_path)
                
                # 确保输出路径的子文件夹存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 将预处理结果安全写入独立的 processed 目录，绝对不污染和覆盖原图
                cv2.imwrite(output_path, result)
                count += 1
                
                if count % 50 == 0 or count == total_files:
                    print(f"⏳ 进度: [{count}/{total_files}] 张图片已完美隔离并保存...")
                    
        print(f"🎉 批量预处理及物理隔离大功告成！成功转移并处理 {count} 张图片。")

# --- 运行批量/单张处理测试 ---
if __name__ == "__main__":
    processor = ImageProcessor()
    
    # 动态获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 🌟 建立绝对干净、各司其职的物理隔离双目录
    raw_folder = os.path.normpath(os.path.join(current_dir, "../../data/raw"))
    processed_folder = os.path.normpath(os.path.join(current_dir, "../../data/processed"))
    
    # 如果本地没有 data/processed 文件夹，代码会自动帮你优雅创建
    os.makedirs(processed_folder, exist_ok=True)
    
    print("="*60)
    print("      Fruit-Freshness-Detector 图像预处理流水线 (最终战神版)")
    print("="*60)
    
    # ⬇️ 一键调用全新的安全批量隔离函数
    processor.process_all_images(raw_folder, processed_folder)