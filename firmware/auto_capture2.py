import requests
import time
import random

ESP32_IP = "192.168.4.1"
INTERVAL = 5  # 拍照间隔（秒），可以改成你想要的数值

def take_photo():
    try:
        # 加随机参数，强制刷新，防止返回旧照片
        url = f"http://{ESP32_IP}/capture?t={random.randint(1, 999999)}"
        print("正在拍照...")
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            filename = f"photo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"✅ 已保存: {filename} (大小: {len(response.content)} 字节)")
            return True
        else:
            print(f"❌ 失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

print(f"开始连续拍照，每 {INTERVAL} 秒一张，按 Ctrl+C 停止")
print("=" * 50)

count = 0
while True:
    count += 1
    print(f"\n第 {count} 次拍照")

    if take_photo():
        pass

    time.sleep(INTERVAL)