import os
import shutil
import random
from PIL import Image

def run_organization():
    # --- 1. 配置路径 ---
    # 原始图片路径 (根据你的实际位置修改)
    src_root = r'E:\Fruit-Freshness-Detector\pict'
    # 目标数据集路径
    dest_root = r'E:\Fruit-Freshness-Detector\algorithm\datasets\fruit_data'
    
    categories = ['tomato', 'strawberry', 'banana', 'apple']
    train_ratio = 0.8

    # --- 2. 创建目录结构 ---
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(dest_root, folder, split), exist_ok=True)

    # --- 3. 开始搬运 ---
    for cat in categories:
        cat_src = os.path.join(src_root, cat)
        if not os.path.exists(cat_src):
            print(f"⚠️ 跳过: 找不到文件夹 {cat_src}")
            continue

        # 获取所有图片文件
        files = [f for f in os.listdir(cat_src) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))]
        random.shuffle(files)
        
        split_idx = int(len(files) * train_ratio)
        print(f"📦 正在处理 {cat}: 共 {len(files)} 张图片...")

        for i, filename in enumerate(files):
            subset = 'train' if i < split_idx else 'val'
            # 命名净化：apple_1.jpg
            new_name = f"{cat}_{i+1}.jpg"
            
            src_file = os.path.join(cat_src, filename)
            dest_file = os.path.join(dest_root, 'images', subset, new_name)

            try:
                # 使用 PIL 转换并保存为 JPG，同时可以统一尺寸（可选）
                with Image.open(src_file) as img:
                    img = img.convert('RGB')
                    # 如果需要缩放，取消下面一行的注释:
                    # img = img.resize((640, 640)) 
                    img.save(dest_file, 'JPEG')
            except Exception as e:
                print(f"❌ 无法处理 {filename}: {e}")

    print("\n✅ 数据集整理完成！你可以开始使用 LabelImg 标注了。")

if __name__ == "__main__":
    run_organization()