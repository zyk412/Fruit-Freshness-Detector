import serial
import time
import csv

SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
OUTPUT_FILE = 'spectral.csv'

print("Connecting to", SERIAL_PORT)
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

print("Connected! Waiting for one set of data...")
print(f"Data will be saved to {OUTPUT_FILE}")
print("Press Ctrl+C to stop waiting")

with open(OUTPUT_FILE, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 表头：10个光谱通道
    writer.writerow(['415nm', '445nm', '480nm', '515nm', '555nm',
                     '590nm', '630nm', '680nm', 'Clear', 'NIR'])

    try:
        while True:
            line = ser.readline()
            if line:
                line_str = line.decode('utf-8').strip()

                if line_str:
                    print("Received:", line_str)

                    parts = line_str.split(',')

                    # 接收10个数值后保存并退出
                    if len(parts) == 10:
                        writer.writerow(parts)
                        csvfile.flush()
                        print(f"\nSuccess! Saved 1 record to {OUTPUT_FILE}")
                        print("Program finished.")
                        break
                    else:
                        print(f"Warning: Expected 10 values, got {len(parts)}. Ignoring...")

    except KeyboardInterrupt:
        print(f"\nCancelled by user. No data saved.")
    finally:
        ser.close()