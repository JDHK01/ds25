#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机端通信程序
用于与地面站进行双向通信，发送动物检测数据，接收控制指令
"""

import socket
import json
import threading
import time
import queue
from datetime import datetime

class DroneComm:
    def __init__(self, ground_ip='192.168.1.100', ground_port=8888):
        self.ground_ip = ground_ip
        self.ground_port = ground_port
        self.socket = None
        self.running = False
        self.send_queue = queue.Queue()
        self.recv_queue = queue.Queue()
        
    def connect(self):
        """连接到地面站"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.ground_ip, self.ground_port))
            self.running = True
            print(f"成功连接到地面站 {self.ground_ip}:{self.ground_port}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self.running = False
        if self.socket:
            self.socket.close()
            print("已断开与地面站的连接")
    
    def send_animal_detection(self, grid_code, animals_data):
        """发送动物检测数据
        Args:
            grid_code: 方格代码，如'A3B5'
            animals_data: 动物数据列表或字典
                - 列表格式: [{'type': '象', 'count': 2}, {'type': '虎', 'count': 1}]
                - 字典格式: {'象': 2, '虎': 1, '猴': 3}
        """
        # 统一转换为列表格式
        if isinstance(animals_data, dict):
            animals_list = [{'type': animal_type, 'count': count} 
                          for animal_type, count in animals_data.items() if count > 0]
        elif isinstance(animals_data, list):
            animals_list = [animal for animal in animals_data if animal.get('count', 0) > 0]
        else:
            # 兼容旧格式：单个动物类型和数量
            if isinstance(animals_data, tuple) and len(animals_data) == 2:
                animal_type, count = animals_data
                animals_list = [{'type': animal_type, 'count': count}]
            else:
                raise ValueError("animals_data 格式不正确")
        
        # 如果没有检测到动物，不发送数据
        if not animals_list:
            return
            
        data = {
            'type': 'animal_detection',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'grid_code': grid_code,
            'animals': animals_list,  # 改为animals列表
            'total_count': sum(animal['count'] for animal in animals_list)
        }
        self.send_queue.put(data)
    
    def send_status(self, status_type, value):
        """发送无人机状态信息
        Args:
            status_type: 状态类型，如'altitude', 'battery', 'position'
            value: 状态值
        """
        data = {
            'type': 'status',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'status_type': status_type,
            'value': value
        }
        self.send_queue.put(data)
    
    def send_mission_complete(self, total_animals):
        """发送任务完成信息
        Args:
            total_animals: 各种动物的总数统计字典
        """
        data = {
            'type': 'mission_complete',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'total_animals': total_animals
        }
        self.send_queue.put(data)
    
    def _send_thread(self):
        """发送线程"""
        while self.running:
            try:
                if not self.send_queue.empty():
                    data = self.send_queue.get()
                    json_data = json.dumps(data, ensure_ascii=False)
                    # 添加消息长度头
                    msg = f"{len(json_data):08d}{json_data}"
                    self.socket.sendall(msg.encode('utf-8'))
                    print(f"发送: {data}")
                time.sleep(0.01)
            except Exception as e:
                print(f"发送错误: {e}")
                self.running = False
    
    def _recv_thread(self):
        """接收线程"""
        buffer = ""
        while self.running:
            try:
                # 接收数据
                data = self.socket.recv(1024).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                # 处理完整的消息
                while len(buffer) >= 8:
                    # 获取消息长度
                    msg_len = int(buffer[:8])
                    
                    # 检查是否接收到完整消息
                    if len(buffer) >= 8 + msg_len:
                        # 提取消息
                        json_data = buffer[8:8+msg_len]
                        buffer = buffer[8+msg_len:]
                        
                        # 解析JSON
                        try:
                            msg = json.loads(json_data)
                            self.recv_queue.put(msg)
                            print(f"接收: {msg}")
                            
                            # 处理指令
                            self._handle_command(msg)
                        except json.JSONDecodeError as e:
                            print(f"JSON解析错误: {e}")
                    else:
                        break
                        
            except Exception as e:
                print(f"接收错误: {e}")
                self.running = False
    
    def _handle_command(self, command):
        """处理接收到的指令"""
        cmd_type = command.get('type')
        
        if cmd_type == 'set_no_fly_zone':
            # 设置禁飞区
            zones = command.get('zones', [])
            print(f"设置禁飞区: {zones}")
            # 这里可以调用实际的禁飞区设置函数
            
        elif cmd_type == 'start_patrol':
            # 开始巡查
            print("收到开始巡查指令")
            # 这里可以调用实际的巡查启动函数
            
        elif cmd_type == 'emergency_stop':
            # 紧急停止
            print("收到紧急停止指令")
            # 这里可以调用实际的紧急停止函数
            
        elif cmd_type == 'return_home':
            # 返航
            print("收到返航指令")
            # 这里可以调用实际的返航函数
    
    def start(self):
        """启动通信线程"""
        if self.connect():
            # 启动发送和接收线程
            send_thread = threading.Thread(target=self._send_thread)
            recv_thread = threading.Thread(target=self._recv_thread)
            
            send_thread.daemon = True
            recv_thread.daemon = True
            
            send_thread.start()
            recv_thread.start()
            
            return True
        return False

# 使用示例
if __name__ == "__main__":
    # 创建通信对象
    drone_comm = DroneComm(ground_ip='127.0.0.1', ground_port=8888)
    
    # 启动通信
    if drone_comm.start():
        # 模拟发送数据
        try:
            # 发送状态信息
            drone_comm.send_status('altitude', 120.5)
            time.sleep(1)
            
            # 发送动物检测数据 - 支持多种格式
            # 方式1：字典格式（同一位置多种动物）
            drone_comm.send_animal_detection('A3B5', {'象': 2, '虎': 1})
            time.sleep(1)
            
            # 方式2：列表格式
            drone_comm.send_animal_detection('B2C4', [{'type': '虎', 'count': 1}, {'type': '猴', 'count': 2}])
            time.sleep(1)
            
            # 方式3：单个动物（兼容旧格式）
            drone_comm.send_animal_detection('C1D3', {'猴': 3})
            time.sleep(1)
            
            # 发送任务完成信息
            total_animals = {
                '象': 2,
                '虎': 1,
                '狼': 0,
                '猴': 3,
                '孔雀': 0
            }
            drone_comm.send_mission_complete(total_animals)
            
            # 保持运行
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n程序中断")
        finally:
            drone_comm.disconnect()