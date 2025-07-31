#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试通信系统
可以在单机上测试地面站和无人机的通信
"""

import sys
import time
import threading
from PyQt5.QtWidgets import QApplication

# 导入地面站
from ground_station import GroundStation

# 导入无人机通信模块
from drone_communication import DroneComm

def run_drone_simulation():
    """模拟无人机发送数据"""
    time.sleep(10)  # 等待地面站启动
    
    # 创建无人机通信对象
    drone = DroneComm(ground_ip='127.0.0.1', ground_port=8888)
    
    if drone.start():
        print("无人机通信已启动")
        
        # 模拟巡查过程
        animals_data = [
            ('A1B1', '象', 1),
            ('A3B2', '虎', 1),
            ('A5B3', '猴', 2),
            ('A7B4', '孔雀', 1),
            ('A2B5', '狼', 1),
            ('A4B6', '象', 1),
            ('A6B7', '猴', 1),
        ]
        
        # 发送初始状态
        drone.send_status('altitude', 120.0)
        drone.send_status('battery', 85)
        time.sleep(1)
        
        # 模拟检测过程
        for grid, animal, count in animals_data:
            drone.send_animal_detection(grid, animal, count)
            time.sleep(2)  # 模拟飞行时间
            
            # 偶尔发送状态更新
            if grid in ['A3B2', 'A6B7']:
                battery = 85 - animals_data.index((grid, animal, count)) * 5
                drone.send_status('battery', battery)
        
        # 发送任务完成
        total = {
            '象': 2,
            '虎': 1,
            '狼': 1,
            '猴': 3,
            '孔雀': 1
        }
        drone.send_mission_complete(total)
        
        print("模拟任务完成")
        
        # 保持连接
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            drone.disconnect()

def main():
    """主函数"""
    print("野生动物巡查系统测试")
    print("=" * 50)
    print("1. 启动地面站")
    print("2. 点击'启动服务器'按钮")
    print("3. 无人机将自动连接并发送模拟数据")
    print("=" * 50)
    
    # 在后台线程运行无人机模拟
    drone_thread = threading.Thread(target=run_drone_simulation)
    drone_thread.daemon = True
    drone_thread.start()
    
    # 启动地面站
    app = QApplication(sys.argv)
    station = GroundStation()
    station.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()