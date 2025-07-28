import serial
import time
import math # 用于数学计算，特别是三角函数
import random # 用于生成随机数

# --- 串口设置 ---
# 根据你的实际连接修改 COM 端口和波特率
# Windows: 'COMx' (例如 'COM3', 'COM4')
# Linux/macOS: '/dev/ttyUSB0' 或 '/dev/ttyACM0' 等
SERIAL_PORT = '/dev/ttyUSB0' # <-- 请修改为你的USB转TTL模块连接的串口端口
BAUD_RATE = 9600

# --- 数据包协议定义 (与STM32的C代码保持一致!) ---
LORA_HEADER_GOODS = "$GOODS,"
LORA_HEADER_X       = "$XDATA,"
LORA_HEADER_Y       = "$YDATA,"
# LORA_HEADER_GOODSID = "$GOODSID," # 新增：GOODSID 包头 - 这个不再需要单独发送了，因为ID会包含在GOODS包中
LORA_PACKET_FOOTER = ",END#"

class SerialPort:
    def __init__(self, port, baudrate=9600, timeout=1):
        """
        初始化串口对象
        :param port: 串口号
        :param baudrate: 波特率
        :param timeout: 超时时间
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        
    def open(self):
        """打开串口"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"成功打开串口: {self.port} 波特率: {self.baudrate}.")
            time.sleep(2)  # 等待串口稳定连接
            return True
        except Exception as e:
            print(f"打开串口失败: {e}")
            return False
    
    def close(self):
        """关闭串口"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("串口已关闭。")
    
    def is_open(self):
        """检查串口是否打开"""
        return self.ser.is_open if self.ser else False
    
    def send_lora_packet(self, header, data_content, footer=LORA_PACKET_FOOTER):
        """
        构造并发送一个完整的LoRa数据包
        :param header: 数据包的包头，例如 "$GOODS,"
        :param data_content: 数据包的实际内容，不包含包头和包尾
        :param footer: 数据包的包尾，默认为"\r\n"
        """
        if not self.ser or not self.ser.is_open:
            print("串口未打开，无法发送数据")
            return False
            
        packet = f"{header}{data_content}{footer}"
        # 将字符串编码为字节，因为串口通信是字节流
        encoded_packet = packet.encode('ascii')  # 假设你的数据是ASCII编码

        print(f"发送: {packet} (编码: {encoded_packet})")
        try:
            self.ser.write(encoded_packet)
            time.sleep(0.05)  # 每次发送后短暂延迟，给接收端一点处理时间
            return True
        except Exception as e:
            print(f"发送数据失败: {e}")
            return False

def main():
    try:
        # 创建串口对象
        ser_port = SerialPort(port=SERIAL_PORT, baudrate=9600)
        
        # 打开串口
        ser_port.open()
        for i in range(1,7):
            # 发送LoRa数据包
            ser_port.send_lora_packet(header=LORA_HEADER_GOODS, data_content=f"{18+i},C{i}")
            time.sleep(1)  # 等待1秒钟再发送下一个包
            
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 确保串口被关闭
        if 'ser_port' in locals():
            ser_port.close()

if __name__ == "__main__":
    main()