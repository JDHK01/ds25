import serial
import time
import threading

# 宏定义 - 配置参数
SERIAL_PORT = 'COM9'
BAUD_RATE = 9600

# 宏定义 - 发送数据包头 (单$)
HEADER_GOODS = '$GOODS,'
HEADER_XDATA = '$XDATA,'
HEADER_YDATA = '$YDATA,'
HEADER_VELOCITY = '$VELOCITY,'
HEADER_SEARCH = '$SEARCH,'
HEADER_COMMAND_TJC = '%COMMAND,'

# 宏定义 - 接收数据包头 ($$)
HEADER_COMMAND_LORA = '$$COMMAND,'

# 宏定义 - 数据包尾
PACKET_FOOTER = ',END#'

# 全局变量
ser = None
rx_buffer = ""

def send_goods_packet(packet_id, goods_number):
    """发送货物数据包"""
    packet = f"{HEADER_GOODS}{packet_id},{goods_number}{PACKET_FOOTER}"
    ser.write(packet.encode('ascii'))
    print(f"发送: {packet}")

def send_xdata_packet(x_coord):
    """发送X坐标数据包"""
    packet = f"{HEADER_XDATA}{x_coord}{PACKET_FOOTER}"
    ser.write(packet.encode('ascii'))
    print(f"发送: {packet}")

def send_ydata_packet(y_coord):
    """发送Y坐标数据包"""
    packet = f"{HEADER_YDATA}{y_coord}{PACKET_FOOTER}"
    ser.write(packet.encode('ascii'))
    print(f"发送: {packet}")

def send_velocity_packet(velocity):
    """发送速度数据包"""
    packet = f"{HEADER_VELOCITY}{velocity}{PACKET_FOOTER}"
    ser.write(packet.encode('ascii'))
    print(f"发送: {packet}")

def send_search_packet(search_id):
    """发送查询数据包"""
    packet = f"{HEADER_SEARCH}{search_id}{PACKET_FOOTER}"
    ser.write(packet.encode('ascii'))
    print(f"发送: {packet}")

def send_command_packet(command):
    """发送TJC命令数据包"""
    packet = f"{HEADER_COMMAND_TJC}{command}{PACKET_FOOTER}"
    ser.write(packet.encode('ascii'))
    print(f"发送: {packet}")

def run_automation_script():
    """运行自动化脚本"""
    print("检测到$$COMMAND数据包，运行自动化脚本...")
    time.sleep(8)
    print("模拟程序开始！")
    # 发送A1-A4的goods包
    for i in range(4):
        send_goods_packet(i, f"A{i+1}")
        time.sleep(1)
    
    print("等待3秒...")
    time.sleep(3)
    
    # 发送查询包
    for i in range(4):
        send_search_packet(i)
        time.sleep(1)
    
    print("自动化脚本执行完毕")

def process_received_packet(packet):
    """处理接收到的数据包"""
    if packet.startswith(HEADER_COMMAND_LORA) and packet.endswith(PACKET_FOOTER):
        # 提取命令内容
        command = packet[len(HEADER_COMMAND_LORA):-len(PACKET_FOOTER)]
        print(f"接收到$$COMMAND数据包: {command}")
        
        # 在新线程中运行脚本，避免阻塞
        script_thread = threading.Thread(target=run_automation_script)
        script_thread.daemon = True
        script_thread.start()
        return True
    return False

def main():
    global ser, rx_buffer
    
    try:
        # 打开串口
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"串口已打开: {SERIAL_PORT} @ {BAUD_RATE}")
        
        print("程序已启动，等待$$COMMAND数据包...")
        print("输入 'quit' 退出程序")
        
        # 主循环
        i = True
        while i:
            # 检查键盘输入
            import select
            import sys
            if sys.platform != 'win32':
                # Linux/Mac系统
                if select.select([sys.stdin], [], [], 0)[0]:
                    user_input = input().strip()
                    if user_input.lower() == 'quit':
                        break
            else:
                # Windows系统 - 简化处理
                import msvcrt
                if msvcrt.kbhit():
                    user_input = input().strip()
                    if user_input.lower() == 'quit':
                        break
            
            # 检查串口接收
            if ser.in_waiting > 0:
                i = False
                data = ser.read(ser.in_waiting).decode('ascii', errors='ignore')
                rx_buffer += data
                
                # 查找完整的$$COMMAND数据包
                while True:
                    start_pos = rx_buffer.find('$$')
                    if start_pos == -1:
                        break
                    
                    end_pos = rx_buffer.find(PACKET_FOOTER, start_pos)
                    if end_pos == -1:
                        break
                    
                    # 提取完整数据包
                    packet = rx_buffer[start_pos:end_pos + len(PACKET_FOOTER)]
                    
                    # 处理数据包
                    if process_received_packet(packet):
                        print(f"已处理数据包: {packet}")
                    
                    # 移除已处理的数据
                    rx_buffer = rx_buffer[end_pos + len(PACKET_FOOTER):]
                
                # 清理缓冲区中的无效数据
                if len(rx_buffer) > 500:  # 防止缓冲区过大
                    rx_buffer = rx_buffer[-200:]  # 保留最后200个字符
            
            time.sleep(0.01)  # 短暂休眠
            
    except serial.SerialException as e:
        print(f"串口错误: {e}")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("串口已关闭")

if __name__ == "__main__":
    main()