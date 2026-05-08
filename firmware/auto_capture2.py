import requests
import time
import random
import os

# 配置信息
ESP32_IP = "192.168.4.1"
INTERVAL = 5  # 拍照间隔（秒）

# 路径配置：指向根目录下的 data/raw
# 因为脚本在 firmware 文件夹内，所以用 .. 飞到上一层，再进入 data/raw
current_file_path = os.path.abspath(__file__) # firmware/auto_capture2.py
firmware_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(firmware_dir)
SAVE_DIR = os.path.join(project_root, "data", "raw")

def take_photo():
    # 自动创建数据保存目录（如果不存在的话）
    if not os.path.exists(SAVE_DIR):
        try:
            os.makedirs(SAVE_DIR)
            print(f"已创建数据目录: {SAVE_DIR}")
        except Exception as e:
            print(f"创建目录失败: {e}")
            return False

    try:
        # 加随机参数，强制刷新，防止返回旧照片
        url = f"http://{ESP32_IP}/capture?t={random.randint(1, 999999)}"
        print("正在触发硬件拍照...")
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            # 生成文件名
            base_filename = f"photo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            # 组合完整路径：../data/raw/photo_xxx.jpg
            full_path = os.path.join(SAVE_DIR, base_filename)
            
            with open(full_path, "wb") as f:
                f.write(response.content)
            
            print(f"✅ 已保存至: {full_path} (大小: {len(response.content)} 字节)")
            return True
        else:
            print(f"❌ 拍摄失败，服务器返回码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 网络请求或保存出错: {e}")
        return False

# 主程序逻辑
if __name__ == "__main__":
    print(f"FruitFresh 项目 - 硬件采集模块启动")
    print(f"保存路径: {os.path.abspath(SAVE_DIR)}")
    print(f"间隔时间: {INTERVAL} 秒/张，按 Ctrl+C 停止")
    print("=" * 50)

    count = 0
    try:
        while True:
            count += 1
            print(f"\n[任务 #{count}] 开始执行...")
            take_photo()
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\n停止拍照，正在退出程序...")