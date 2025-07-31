import serial
import time
import threading

# --- 串口设置 ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600

# --- 发送数据包协议定义 ---
LORA_HEADER_GOODS = "$GOODS,"
LORA_HEADER_X = "$XDATA,"
LORA_HEADER_Y = "$YDATA,"
LORA_HEADER_VELOCITY = "$VELOCITY,"
LORA_HEADER_SEARCH = "$SEARCH,"
LORA_HEADER_SEARCHID = "$$SEARCHID,"
LORA_HEADER_COMMAND_TJC = "%COMMAND,"
REICEIVE = '#'

# --- 接收数据包协议定义 ---
LORA_HEADER_COMMAND_RECV = "$$COMMAND,"

# --- 通用包尾 ---
LORA_PACKET_FOOTER = "%"

class SerialPort:
    def __init__(self, port, baudrate=9600, timeout=0.2):
        """
        初始化串口对象
        :param port: 串口号
        :param baudrate: 波特率
        :param timeout: 超时时间
        """
        # 串口的相关信息
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.rx_buffer = ""
        self.receivetime = 0

        # 为接收做的处理
        self.receive_thread = None
        self.is_receiving = False

        # 处理函数
        self.packet_handlers = {}  # 存储不同包头的处理函数

        
    def open(self):
        """打开串口"""
        try:
            # 创建串口对象
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"成功打开串口: {self.port} 波特率: {self.baudrate}.")
            time.sleep(2)  # 等待串口稳定连接
            return True
        except Exception as e:
            print(f"打开串口失败: {e}")
            return False
    
    def close(self):
        """关闭串口"""
        # 关闭接收标识位+ 关闭串口
        self.stop_receiving()
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("串口已关闭。")
    
    def is_open(self):
        """检查串口是否打开"""
        return self.ser.is_open if self.ser else False
    
    def send_lora_packet(self, header, data_content, footer=LORA_PACKET_FOOTER):
        """
        构造并发送一个完整的LoRa数据包
        :param header: 数据包的包头
        :param data_content: 数据包的实际内容
        :param footer: 数据包的包尾
        """
        if not self.ser or not self.ser.is_open:
            print("串口未打开，无法发送数据")
            return False
        # 包的内容:头 + 数据内容 + 尾
        # 编码包
        # 发送
        packet = f"{header}{data_content}{footer}"
        encoded_packet = packet.encode('ascii')

        print(f"发送: {packet}")
        try:
            self.ser.write(encoded_packet)
            time.sleep(0.1)
            return True
        except Exception as e:
            print(f"发送数据失败: {e}")
            return False
    
    # --- 发送便捷方法 ---
    def send_goods_packet(self, packet_id, goods_number):
        """发送货物数据包"""
        return self.send_lora_packet(LORA_HEADER_GOODS, f"{packet_id},{goods_number}")
    
    def send_xdata_packet(self, x_coord):
        """发送X坐标数据包"""
        return self.send_lora_packet(LORA_HEADER_X, str(x_coord))
    
    def send_ydata_packet(self, y_coord):
        """发送Y坐标数据包"""
        return self.send_lora_packet(LORA_HEADER_Y, str(y_coord))
    
    def send_velocity_packet(self, velocity):
        """发送速度数据包"""
        return self.send_lora_packet(LORA_HEADER_VELOCITY, str(velocity))
    
    def send_search_packet(self, search_id):
        """发送查询数据包"""
        return self.send_lora_packet(LORA_HEADER_SEARCH, str(search_id))
    
    def send_command_packet(self, command):
        """发送命令数据包"""
        return self.send_lora_packet(LORA_HEADER_COMMAND_TJC, str(command))
    
    def register_packet_handler(self, header, handler):
        """
        注册数据包处理函数
        :param header: 数据包头
        :param handler: 处理函数，接收解析后的数据作为参数
        """
        self.packet_handlers[header] = handler
        print(f"已注册处理器: {header}")
    
    def start_receiving(self):
        """开始在后台线程接收数据"""
        if self.is_receiving:
            print("已在接收数据")
            return
        self.is_receiving = True
        self.receive_thread = threading.Thread(target=self._receive_loop)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        print("开始接收数据...")
    
    def stop_receiving(self):
        self.is_receiving = False
        if self.receive_thread:
            self.receive_thread.join(timeout=1)# 让这个线程在timeout时间内停止
            print("停止接收数据")
    
    def _receive_loop(self):
        """接收循环（在后台线程运行）"""
        while self.is_receiving and self.ser and self.ser.is_open:
            try:
                # 检查是否有数据可读
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting).decode('ascii', errors='ignore')
                    self.rx_buffer += data
                    
                    # 处理缓冲区中的完整数据包
                    self._process_buffer()
                    
                time.sleep(0.01)  # 短暂休眠，避免占用过多CPU
                
            except Exception as e:
                print(f"接收数据时出错: {e}")
    
    def _process_buffer(self):
        """处理接收缓冲区中的数据包"""
        while True:
            # 查找数据包开始位置（可能是$或$$或#）
            packet_start = -1
            header_found = None
            
            # 检查所有已注册的包头
            for header in self.packet_handlers.keys():
                pos = self.rx_buffer.find(header)
                if pos != -1 and (packet_start == -1 or pos < packet_start):
                    packet_start = pos
                    header_found = header
            
            if packet_start == -1:
                # 没有找到任何包头
                break
            
            # 根据包头选择包尾
            if header_found == "#":
                packet_footer = "%"
            else:
                packet_footer = LORA_PACKET_FOOTER
                
            # 查找包尾
            packet_end = self.rx_buffer.find(packet_footer, packet_start)
            if packet_end == -1:
                # 包尾还没接收到，等待更多数据
                break
            
            # 提取完整数据包
            packet = self.rx_buffer[packet_start:packet_end + len(packet_footer)]
            self.receivetime = 1
            # 解析数据内容
            content_start = packet_start + len(header_found)
            content_end = packet_end
            content = self.rx_buffer[content_start:content_end]
            
            # 调用对应的处理函数
            if header_found in self.packet_handlers:
                try:
                    self.packet_handlers[header_found](content, packet)
                except Exception as e:
                    print(f"处理数据包时出错: {e}")
            self.receivetime += 1
            # 从缓冲区移除已处理的数据
            self.rx_buffer = self.rx_buffer[packet_end + len(packet_footer):]

        # 防止缓冲区过大
        if len(self.rx_buffer) > 500:
            self.rx_buffer = self.rx_buffer[-200:]

# --- 示例：数据包处理函数 ---
def handle_command_packet(content, full_packet):
    """处理$$COMMAND数据包"""
    print(f"接收到COMMAND数据包: {content}")
    print(f"完整数据包: {full_packet}")
    
    # 这里可以添加具体的处理逻辑
    # 比如根据命令内容执行相应操作

# --- 主程序示例 ---
def main():
    try:
        # 创建串口对象
        ser_port = SerialPort(port=SERIAL_PORT, baudrate=BAUD_RATE)
        if not ser_port.open():
            return
        


        ser_port.send_goods_packet(1, f"D{1}")
        time.sleep(1)
        ser_port.send_goods_packet(2, f"C{1}")
        time.sleep(1)
        ser_port.send_goods_packet(3, f"B{1}")
        time.sleep(1)
        ser_port.send_goods_packet(4, f"A{1}")
        time.sleep(1)


        # 保持程序运行，等待接收数据
        def command_handler(content, full_packet):
            print(f"收到命令: {content}")
        ser_port.register_packet_handler(LORA_HEADER_COMMAND_RECV, command_handler)
        ser_port.register_packet_handler(REICEIVE, command_handler)
        ser_port.register_packet_handler(LORA_HEADER_SEARCHID, command_handler)
        ser_port.send_lora_packet(LORA_HEADER_SEARCH ,1 , footer=LORA_PACKET_FOOTER)
        ser_port.start_receiving()
        print("\n等待接收数据... (按Ctrl+C退出)")
        while ser_port.receivetime < 1:
            time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 确保串口被关闭
        if 'ser_port' in locals():
            ser_port.close()

if __name__ == "__main__":
    main()