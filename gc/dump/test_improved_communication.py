#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进后的通信协议
验证同一位置多种动物检测的数据结构和通信
"""

import time
import threading
from drone_communication import DroneComm

def test_multiple_animals_detection():
    """测试多种动物检测数据发送"""
    print("=" * 50)
    print("测试改进后的动物检测通信协议")
    print("=" * 50)
    
    # 创建通信对象
    drone_comm = DroneComm(ground_ip='127.0.0.1', ground_port=8888)
    
    # 启动通信
    if drone_comm.start():
        try:
            print("连接成功，开始测试...")
            time.sleep(1)
            
            # 测试1：字典格式 - 同一位置多种动物
            print("\n测试1：字典格式（同一位置多种动物）")
            animals_dict = {'象': 2, '虎': 1, '猴': 3}
            drone_comm.send_animal_detection('A3B5', animals_dict)
            print(f"发送数据：位置A3B5，动物{animals_dict}")
            time.sleep(2)
            
            # 测试2：列表格式 - 详细动物信息
            print("\n测试2：列表格式")
            animals_list = [
                {'type': '虎', 'count': 1},
                {'type': '狼', 'count': 2}, 
                {'type': '孔雀', 'count': 5}
            ]
            drone_comm.send_animal_detection('B4C2', animals_list)
            print(f"发送数据：位置B4C2，动物{animals_list}")
            time.sleep(2)
            
            # 测试3：空数据（应该不发送）
            print("\n测试3：空数据测试")
            empty_dict = {'象': 0, '虎': 0}
            drone_comm.send_animal_detection('C1D3', empty_dict)
            print(f"发送数据：位置C1D3，动物{empty_dict}（应该不发送）")
            time.sleep(2)
            
            # 测试4：单个动物（向后兼容）
            print("\n测试4：单个动物（向后兼容）")
            single_animal = {'猴': 4}
            drone_comm.send_animal_detection('D2E1', single_animal)
            print(f"发送数据：位置D2E1，动物{single_animal}")
            time.sleep(2)
            
            # 测试5：混合格式
            print("\n测试5：大规模混合检测")
            complex_detection = {
                '象': 3,
                '虎': 2, 
                '狼': 1,
                '猴': 8,
                '孔雀': 12
            }
            drone_comm.send_animal_detection('A1B1', complex_detection)
            print(f"发送数据：位置A1B1，动物{complex_detection}")
            time.sleep(2)
            
            # 发送任务完成信息
            print("\n发送任务完成统计...")
            total_animals = {
                '象': 5,
                '虎': 4,
                '狼': 3,
                '猴': 15,
                '孔雀': 17
            }
            drone_comm.send_mission_complete(total_animals)
            print(f"任务完成统计：{total_animals}")
            
            print("\n测试完成！请检查地面站接收情况。")
            
            # 保持连接一段时间
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n测试中断")
        finally:
            drone_comm.disconnect()
            print("断开连接")

def test_data_structure_validation():
    """测试数据结构验证"""
    print("\n" + "=" * 50)
    print("测试数据结构验证")
    print("=" * 50)
    
    drone_comm = DroneComm()
    
    # 测试各种数据格式
    test_cases = [
        # 有效格式
        ({'象': 2, '虎': 1}, "字典格式"),
        ([{'type': '象', 'count': 2}], "列表格式"),
        (('象', 2), "元组格式（兼容）"),
        
        # 边界情况
        ({}, "空字典"),
        ([], "空列表"),
        ({'象': 0}, "零数量"),
        ([{'type': '象', 'count': 0}], "零数量列表"),
    ]
    
    for animals_data, description in test_cases:
        try:
            print(f"\n测试 {description}: {animals_data}")
            # 这里只测试数据结构处理，不实际发送
            if isinstance(animals_data, dict):
                animals_list = [{'type': animal_type, 'count': count} 
                              for animal_type, count in animals_data.items() if count > 0]
            elif isinstance(animals_data, list):
                animals_list = [animal for animal in animals_data if animal.get('count', 0) > 0]
            elif isinstance(animals_data, tuple) and len(animals_data) == 2:
                animal_type, count = animals_data
                animals_list = [{'type': animal_type, 'count': count}] if count > 0 else []
            else:
                animals_list = []
            
            print(f"  处理结果: {animals_list}")
            print(f"  是否发送: {'是' if animals_list else '否'}")
            
        except Exception as e:
            print(f"  错误: {e}")

if __name__ == "__main__":
    print("改进后的动物检测通信协议测试")
    print("请确保地面站程序正在运行...")
    
    # 询问是否开始测试
    input("按回车键开始测试...")
    
    # 运行数据结构验证测试
    test_data_structure_validation()
    
    # 运行通信测试
    test_multiple_animals_detection()