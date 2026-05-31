import os
import shutil
import random
from PIL import Image

def run_organization():
    # --- 1. 【核心修改：动态路径计算】 ---
    # 获取当前脚本的绝对路径 (假设脚本在 algorithm/fruit_detection 目录下)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 根据你的项目目录结构，向上回退两层找到项目根目录 E:\Fruit-Freshness-Detector
    # 如果脚本就在根目录下，把下一行改成 project_root = current_dir 即可
    project_root = os.path.normpath(os.path.join(current_dir, "../../"))
    
    # 动态拼接三个核心文件夹路径
    src_web_root = os.path.join(project_root, "pict")               # 网络高清数据集
    src_real_root = os.path.join(project_root, "data", "raw")        # 树莓派实拍数据集
    dest_root = os.path.join(project_root, "algorithm", "datasets", "fruit_data") # 最终YOLO数据集

    # 打印路径信息，方便你运行前核对
    print(f"📁 项目根目录: {project_root}")
    print(f"🌐 网络数据源: {src_web_root}")
    print(f"📸 实拍数据源: {src_real_root}")
    print(f"🎯 最终输出地: {dest_root}\n" + "="*50)

    # --- 2. 【过滤类别】舍弃草莓，只保留我们要融合的三种水果 ---
    categories = ['tomato', 'banana', 'apple']
    train_ratio = 0.8

    # --- 3. 创建标准的 YOLO 目录结构 ---
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(dest_root, folder, split), exist_ok=True)

    # --- 4. 开始双源数据搬运与合并 ---
    for cat in categories:
        all_combined_files = [] # 存储该类别下合并后的“全家福”图片绝对路径
        
        # 【数据源 A】从网络数据集 (pict) 里搜刮
        cat_web_src = os.path.join(src_web_root, cat)
        if os.path.exists(cat_web_src):
            web_files = [os.path.join(cat_web_src, f) for f in os.listdir(cat_web_src) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))]
            all_combined_files.extend(web_files)
            print(f"🔍 [网络源] 加载 {cat} 图片: {len(web_files)} 张")
        else:
            print(f"⚠️ 提示: 找不到网络数据源文件夹 {cat_web_src}")

        # 【数据源 B】从树莓派实拍数据集 (data/raw) 里搜刮
        cat_real_src = os.path.join(src_real_root, cat)
        if os.path.exists(cat_real_src):
            real_files = [os.path.join(cat_real_src, f) for f in os.listdir(cat_real_src) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))]
            all_combined_files.extend(real_files)
            print(f"🔍 [实拍源] 加载 {cat} 图片: {len(real_files)} 张")
        else:
            print(f"⚠️ 提示: 找不到实拍数据源文件夹 {cat_real_src}")

        # 如果两个源加起来都是空的，直接跳过
        if not all_combined_files:
            print(f"❌ 警告: 没有任何关于 {cat} 的图片，跳过该类别。\n")
            continue

        # 混合大池子！打乱两部分数据的顺序，确保训练集和验证集里都有网络图和实拍图
        random.shuffle(all_combined_files)
        
        total_count = len(all_combined_files)
        split_idx = int(total_count * train_ratio)
        print(f"📦 [融合完毕] {cat} 总计: {total_count} 张照片 -> 训练集: {split_idx}张, 验证集: {total_count - split_idx}张")

        # 处理并分发最终的混合图片
        for i, src_file_path in enumerate(all_combined_files):
            subset = 'train' if i < split_idx else 'val'
            
            # 命名净化：彻底规避括号导致的命名冲突，统一变成 类别_序号.jpg
            new_name = f"{cat}_{i+1}.jpg"
            dest_file = os.path.join(dest_root, 'images', subset, new_name)

            try:
                with Image.open(src_file_path) as img:
                    img = img.convert('RGB')
                    # 建议取消下面的注释，将混合图片统一缩放到640x640，训练速度和效果会更好
                    img = img.resize((640, 640), Image.Resampling.LANCZOS) 
                    img.save(dest_file, 'JPEG')
            except Exception as e:
                print(f"❌ 无法处理图片 {os.path.basename(src_file_path)}: {e}")
        print(f"✨ 类别 【{cat}】 组织完毕！\n" + "-"*30)

    print("\n🎉 超级混合数据集构建大功告成！你可以放心去跑 train_cls.py 训练新模型了！")

if __name__ == "__main__":
    run_organization()