import serial
import time

# --- 串口设置 ---
# 根据你的实际连接修改 COM 端口和波特率
# Windows: 'COMx' (例如 'COM3', 'COM4')
# Linux/macOS: '/dev/ttyUSB0' 或 '/dev/ttyACM0' 等
SERIAL_PORT = '/dev/ttyUSB0'  # <-- 请修改为你的STM32连接的串口端口
BAUD_RATE = 9600

# --- 数据包协议定义 (与C代码保持一致!) ---
LORA_HEADER_GOODS = "$GOODS,"
LORA_HEADER_X     = "$XDATA,"
LORA_HEADER_Y     = "$YDATA,"
LORA_PACKET_FOOTER = ",END#"

def send_lora_packet(ser, header, data_content):
    """
    构造并发送一个完整的LoRa数据包。
    """
    packet = f"{header}{data_content}{LORA_PACKET_FOOTER}"
    # 将字符串编码为字节，因为串口通信是字节流
    encoded_packet = packet.encode('ascii') # 假设你的数据是ASCII
    
    print(f"Sending: {packet} (Encoded: {encoded_packet})")
    ser.write(encoded_packet)
    time.sleep(0.1) # 稍微延迟，确保数据完全发出

def send_series(goods_data='1', x_data='2', y_data='3', long_goods_data='4'):
    try:
        # 打开串口
        # timeout=1 可以设置一个读取超时时间，这里我们主要用于写入
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) 
        print(f"Successfully opened serial port: {SERIAL_PORT} at {BAUD_RATE} baud.")
        time.sleep(2) # 等待串口稳定

        # --- 发送不同类型的数据包示例 ---

        # 发送 GOODS 数据
        # goods_data = "Apple:10kg,Banana:5kg"
        print("\n--- Sending GOODS Data ---")
        send_lora_packet(ser, LORA_HEADER_GOODS, goods_data)
        time.sleep(1) # 等待STM32处理

        # 发送 X 坐标数据
        # x_data = "123.45"
        print("\n--- Sending X-coordinate Data ---")
        send_lora_packet(ser, LORA_HEADER_X, x_data)
        time.sleep(1) # 等待STM32处理

        # 发送 Y 坐标数据
        # y_data = "67.89"
        print("\n--- Sending Y-coordinate Data ---")
        send_lora_packet(ser, LORA_HEADER_Y, y_data)
        time.sleep(1) # 等待STM32处理

        # 发送一个更长的 GOODS 数据
        # long_goods_data = "Orange:20kg,Grape:15kg,Pear:12kg,Pineapple:8kg"
        print("\n--- Sending Long GOODS Data ---")
        send_lora_packet(ser, LORA_HEADER_GOODS, long_goods_data)
        time.sleep(1)

        # 模拟连续发送（测试缓冲区和超时解析）
        print("\n--- Sending Rapid Data Burst (GOODS) ---")
        for i in range(5):
            burst_data = f"Item_{i+1}:{(i+1)*10}pcs"
            send_lora_packet(ser, LORA_HEADER_GOODS, burst_data)
            # time.sleep(0.05) # 极短的间隔，测试STM32的解析能力
        time.sleep(2)


    except serial.SerialException as e:
        print(f"Error opening or communicating with serial port: {e}")
        print("Please check if the port is correct and not in use by another application.")
        print("Possible solutions:")
        print("1. Ensure STM32 is connected and drivers are installed.")
        print("2. Close any other serial monitoring software (e.g., STM32CubeMonitor, another serial assistant).")
        print(f"3. Verify the correct port for your device (e.g., in Device Manager on Windows). Current setting: {SERIAL_PORT}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    send_series('1', '2', '3' ,'4')
    
