#!/usr/bin/env python3

"""
边飞行边检测的无人机控制系统
- 按照预定义路径飞行
- 实时进行目标检测
- 检测到目标时暂停飞行，执行视觉导航
- 视觉导航完成后恢复飞行路径
"""


import sys
sys.path.append("/home/by/ds25/temp/mycontrol")
sys.path.append("/home/by/ds25/temp/vision/cv")
sys.path.append("/home/by/ds25/temp/vision/yolo")

sys.path.append("/home/by/ds25/temp/lib")
sys.path.append("/home/by/ds25/temp/gc")
from detect import *
import plan_pro_max
from current_position import *
from flightpath import *
from control import *
from mono_camera import *
from detect_manager import *
from generate_system import *
from ser import * 

import asyncio
import time
import threading
import queue
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
from datetime import datetime
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw, VelocityBodyYawspeed)
from mavsdk.telemetry import LandedState

# ======================预留的信息获取接口===================
def send_detection_result(waypoint_name: str, detections: list, detection_manager: DetectionManager) -> dict:
    """
    发送检测结果
    Args:
        waypoint_name: 当前航点名称
        detections: 检测结果列表
        detection_manager: 检测管理器
        
    Returns:
        包含位置和检测结果的字典
    """
    try:
        # 格式化结果
        # counts = detection_manager.get_detection_counts(detections)
        formatted_result = detection_manager.format_detection_result(detections)
        # 汇总数据
        message = {
            'position': waypoint_name,           # 例如: "A1B1"
            'detections': formatted_result,      # 例如: "{e1m2p3w0t1}"
            'timestamp': datetime.now().isoformat()# 预留接口, 丰富功能
        }
        # 汇总的信息发送接口
        print(f"发送检测结果: 位置={message['position']}, 检测={message['detections']}")
        return message
    except Exception as e:
        print(f"❌ 发送检测结果时出错: {e}")

# 使用优化的航点管理器 (预存储所有坐标，O(1)查找，单次处理保证)
WAYPOINT_MANAGER = OPTIMIZED_WAYPOINT_MANAGER
COORDINATES = WAYPOINT_MANAGER.coordinates  # 向后兼容
DURATION = 3
HEIGHT = -1.1

class WaypointListener:
    """监听航点消息的类"""
    def __init__(self, serial_port='/dev/ttyUSB0', baud_rate=9600):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser_port = None
        self.waypoint_queue = queue.Queue()
        self.is_running = False
    
    def handle_waypoint_message(self, content, full_packet):
        """处理航点消息"""
        try:
            print(f"收到航点消息: {content}")
            waypoint_list = content.split(',')
            waypoint_list = [wp.strip() for wp in waypoint_list if wp.strip()]
            
            if waypoint_list:
                self.waypoint_queue.put(waypoint_list)
                print(f"航点已加入队列: {waypoint_list}")
            else:
                print("收到空航点列表")
        except Exception as e:
            print(f"处理航点消息错误: {e}")
    
    def start_listening(self):
        """开始监听串口消息"""
        try:
            self.ser_port = SerialPort(port=self.serial_port, baudrate=self.baud_rate)
            if not self.ser_port.open():
                print(f"无法打开串口 {self.serial_port}")
                return False
            
            self.ser_port.register_packet_handler("#", self.handle_waypoint_message)
            self.ser_port.start_receiving()
            self.is_running = True
            print(f"开始监听串口 {self.serial_port} 的航点消息...")
            return True
        except Exception as e:
            print(f"启动串口监听失败: {e}")
            return False
    
    def stop_listening(self):
        """停止监听串口消息"""
        if self.ser_port:
            self.ser_port.close()
            self.is_running = False
            print("停止监听串口消息")
    
    def get_waypoints(self, timeout=None):
        """获取航点列表"""
        try:
            return self.waypoint_queue.get(timeout=timeout)
        except queue.Empty:
            return None 

# 效果预览
{
# A9B1: (0.0, 0.0)
# A8B1: (0.5, 0.0)
# A7B1: (1.0, 0.0)
# A6B1: (1.5, 0.0)
# A5B1: (2.0, 0.0)
# A4B1: (2.5, 0.0)
# A3B1: (3.0, 0.0)
# A2B1: (3.5, 0.0)
# A1B1: (4.0, 0.0)
# A9B2: (0.0, 0.5)
# A8B2: (0.5, 0.5)
# A7B2: (1.0, 0.5)
# A6B2: (1.5, 0.5)
# A5B2: (2.0, 0.5)
# A4B2: (2.5, 0.5)
# A3B2: (3.0, 0.5)
# A2B2: (3.5, 0.5)
# A1B2: (4.0, 0.5)
# A9B3: (0.0, 1.0)
# A8B3: (0.5, 1.0)
# A7B3: (1.0, 1.0)
# A6B3: (1.5, 1.0)
# A5B3: (2.0, 1.0)
# A4B3: (2.5, 1.0)
# A3B3: (3.0, 1.0)
# A2B3: (3.5, 1.0)
# A1B3: (4.0, 1.0)
# A9B4: (0.0, 1.5)
# A8B4: (0.5, 1.5)
# A7B4: (1.0, 1.5)
# A6B4: (1.5, 1.5)
# A5B4: (2.0, 1.5)
# A4B4: (2.5, 1.5)
# A3B4: (3.0, 1.5)
# A2B4: (3.5, 1.5)
# A1B4: (4.0, 1.5)
# A9B5: (0.0, 2.0)
# A8B5: (0.5, 2.0)
# A7B5: (1.0, 2.0)
# A6B5: (1.5, 2.0)
# A5B5: (2.0, 2.0)
# A4B5: (2.5, 2.0)
# A3B5: (3.0, 2.0)
# A2B5: (3.5, 2.0)
# A1B5: (4.0, 2.0)
# A9B6: (0.0, 2.5)
# A8B6: (0.5, 2.5)
# A7B6: (1.0, 2.5)
# A6B6: (1.5, 2.5)
# A5B6: (2.0, 2.5)
# A4B6: (2.5, 2.5)
# A3B6: (3.0, 2.5)
# A2B6: (3.5, 2.5)
# A1B6: (4.0, 2.5)
# A9B7: (0.0, 3.0)
# A8B7: (0.5, 3.0)
# A7B7: (1.0, 3.0)
# A6B7: (1.5, 3.0)
# A5B7: (2.0, 3.0)
# A4B7: (2.5, 3.0)
# A3B7: (3.0, 3.0)
# A2B7: (3.5, 3.0)
# A1B7: (4.0, 3.0)
# A9A9: (0.0, 0.0)# 已经遍历完所有航点的标志
}

# 处理提供的航点列表的函数
def create_waypoint_flight_plan(waypoint_names: List[str], height: float = HEIGHT, duration: float = DURATION) -> FlightPathManager:
    """
    根据用户提供的航点名称列表创建飞行计划 (使用优化的航点管理器)
    
    Args:
        waypoint_names: 航点名称列表，如 ["A1B1", "A2B2"]
        height: 飞行高度
        duration: 在每个航点的停留时间
    Returns:
        配置好的FlightPathManager
    """
    
    # 使用优化航点管理器验证航点列表
    validation_result = WAYPOINT_MANAGER.batch_validate_waypoints(waypoint_names)
    
    # 输出验证结果
    if validation_result['invalid_waypoints']:
        print(f"⚠️  发现无效航点: {validation_result['invalid_waypoints']}")
    if validation_result['already_processed']:
        print(f"ℹ️  发现已处理航点: {validation_result['already_processed']}")
    if validation_result['completion_markers']:
        print(f"🏁 发现任务完成标记: {validation_result['completion_markers']}")
    
    # 使用有效航点创建飞行计划
    valid_waypoints = validation_result['valid_waypoints']
    print(f"✅ 有效航点数量: {len(valid_waypoints)}/{len(waypoint_names)}")
    
    # 创建飞行路径管理器
    flight_manager = FlightPathManager()
    flight_manager.create_waypoints_from_user_format(
        waypoint_names=valid_waypoints,  # 只使用有效航点
        coordinate_dict=COORDINATES,
        height=height,
        yaw=0.0,
        duration=duration
    )
    return flight_manager

# 处理检测到物体时的逼近逻辑
async def approach_detected_objects(drone, vision_system: VisionGuidanceSystem, 
                                  detection_manager: DetectionManager, 
                                  waypoint_name: str, waypoint_position: Tuple[float, float, float]):
    """
    处理检测到物体时的逼近逻辑
    
    Args:
        drone: 无人机对象
        vision_system: 视觉导航系统
        detection_manager: 检测管理器
        waypoint_name: 当前航点名称
        waypoint_position: 当前航点位置
    """
    print(f"🎯 在航点 {waypoint_name} 开始物体逼近程序")
    
    # 获取当前帧的检测结果
    detections = detection_manager.detect_objects_from_camera(waypoint_name, waypoint_position)
    
    if not detections:
        print("未检测到物体，继续下一个航点")
        return
    
    # 显示检测统计信息
    counts = detection_manager.get_detection_counts(detections)
    formatted_result = detection_manager.format_detection_result(detections)
    print(f"检测到 {len(detections)} 个物体: {counts}")
    print(f"格式化结果: {formatted_result}")
    print("开始逐个逼近")
    
    # 对每个检测到的物体进行逼近
    for i, detection in enumerate(detections, 1):
        print(f"\n🔍 开始逼近第 {i} 个物体: {detection.class_name} (置信度: {detection.confidence:.2f})")
        
        # 重置视觉导航系统状态
        vision_system.reset_task()
        
        try:
            # 执行视觉导航逼近
            # await drone_control_loop(vision_system, drone)
            print("已经取消了逼近")
            print(f"✅ 成功逼近物体 {i}: {detection.class_name}")
            
        except Exception as e:
            print(f"❌ 逼近物体 {i} 时发生错误: {e}")
            continue
        
        # 短暂停留后继续下一个物体, 稳定后可以删除
        await asyncio.sleep(1.0)
    
    print(f"🏁 航点 {waypoint_name} 的所有物体逼近完成")

# 运行
async def run():
    # ====================接收串口屏的消息=======================
    DRONERECEIVE = '#'
    DRONESEND = '$ANI'
    LORA_PACKET_FOOTER = "%"
    # --- 串口设置 ---
    SERIAL_PORT = '/dev/ttyUSB0'
    BAUD_RATE = 9600

    ser_port = SerialPort(port=SERIAL_PORT, baudrate=BAUD_RATE)
    if not ser_port.open():
        return
    
    # 保持程序运行，等待接收数据
    def command_handler(content, full_packet):
        # ----------------------------------解析字符串----------------------------
        global mylist
        mylist = content.split(',')
        print(f"收到命令: {content}")
        print(mylist)

    ser_port.register_packet_handler(DRONERECEIVE, command_handler)
    ser_port.start_receiving()

    print("等待接收数据")
    while ser_port.receivetime < 1:
        time.sleep(0.1)
    print('收到禁飞区:')
    print(mylist)
    routine = plan_pro_max.get_mapping_result(tuple(sorted(mylist)))
    print('使用的航点清单:')
    print(routine)

    # 发送的api
    ser_port.send_lora_packet(DRONESEND ,'A1B1' + format_animal_counts(result), footer=LORA_PACKET_FOOTER)
    # ====================点亮小灯作为指示灯:防止发生意外=====================





    # ==================== 无人机初始化 ====================
    drone = System()
    await drone.connect(system_address="udp://127.0.0.1:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for drone to have a local position estimate and be armable...")
    async for health in drone.telemetry.health():
        if health.is_local_position_ok and health.is_armable:
            print("-- Local position estimate OK and drone is armable")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Setting initial setpoint")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 90.0))
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))

    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed with error code: {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return

    # ======================================== 配置系统 ============================
    # 相机配置
    camera_config = CameraConfig(
        width=640,
        height=480,
        fps=30,
        device_id=0,
        offset_forward=0.0,# 待调
        offset_right=0.0,# 待调
        offset_down=0.0,
        show_window=False
    )
    
    # 导航配置
    navigation_config = {
        # -------------------------实际测试像素长度宽度和面积, 便于测试---------------------------------
        # ------------------------设置对准的时间--------------------------
        'position_tolerance': 100,
        'min_target_area': 1000,# 待调
        'max_velocity': 0.5,
        'offset_compensation_gain': 0.7,# 待调
        'alignment_duration': 0.5,# 待调
        'completion_tolerance': 80# 待调
    }
    
    # PID配置
    #---------------------------------待调-------------------------
    pid_config = {
        'horizontal': {'kp': 0.1, 'ki': 0.0, 'kd': 0.0, 'output_limit': 0.5},
        'vertical': {'kp': 0.1, 'ki': 0.0, 'kd': 0.0, 'output_limit': 0.5},
        'forward': {'kp': 0.1, 'ki': 0.0, 'kd': 0.0, 'output_limit': 0.3}
    }
    
    # 创建视觉导航系统
    vision_system = VisionGuidanceSystem(
        camera_config=camera_config,
        target_mode=TargetMode.DOWN,
        navigation_config=navigation_config,
        pid_config=pid_config
    )
    
    detection_manager = DetectionManager(
        model_path="vision/yolo/best9999.engine",
        conf_threshold=0.5,
        device="cuda",  # 或者 "cuda" 如果有GPU
        camera_id=0
    )
    
    # ==================== 定义飞行路径 ====================
    flight_manager = create_waypoint_flight_plan(routine, HEIGHT, DURATION)
    print(f"飞行路径规划完成，共 {len(flight_manager.waypoints)} 个航点")
    # 开始任务
    flight_manager.start_mission()
    
    try:
        # ==================== 主飞行循环 ====================
        while not flight_manager.is_completed():
            current_waypoint = flight_manager.get_current_waypoint()
            if not current_waypoint:
                break
            print(f"\n🛩️  前往航点: {current_waypoint.name} at ({current_waypoint.north:.1f}, {current_waypoint.east:.1f}, {current_waypoint.down:.1f})")
            # -------------------------------等待替换-----------------------------------------
            await goto_position_ned(
                drone, 
                current_waypoint.north, 
                current_waypoint.east, 
                current_waypoint.down, 
                current_waypoint.yaw, 
                current_waypoint.duration
            )
            
            # 标记航点到达
            flight_manager.mark_waypoint_arrived()
            
            # ==================== 航点检测和逼近 (优化：每个航点只检测一次) ====================
            if current_waypoint.enable_detection:
                # 检查航点是否应该处理 (防止重复图像识别)
                if not WAYPOINT_MANAGER.should_process_waypoint(current_waypoint.name):
                    print(f"⏭️  航点 {current_waypoint.name} 已处理过，跳过检测")
                    # 航点已处理过，直接跳过所有检测相关处理
                else:
                    print(f"🔍 在航点 {current_waypoint.name} 开始物体检测 (首次处理)")
                    
                    # 进入检测模式
                    flight_manager.enter_detection_mode()
                    
                    # 从摄像头检测物体
                    waypoint_position = (current_waypoint.north, current_waypoint.east, current_waypoint.down)
                    detections = detection_manager.detect_objects_from_camera(
                        current_waypoint.name, 
                        waypoint_position
                    )
                    
                    # 标记航点已处理 (确保只处理一次)
                    WAYPOINT_MANAGER.mark_waypoint_processed(current_waypoint.name)
                    
                    # 保存检测信息
                    detection_manager.save_detection_info(
                        current_waypoint.name, 
                        waypoint_position, 
                        detections
                    )
                    
                    # 处理检测结果
                    if detections:
                        counts = detection_manager.get_detection_counts(detections)
                        formatted_result = detection_manager.format_detection_result(detections)
                        print(f"🎯 航点 {current_waypoint.name} 检测统计: {counts}")
                        print(f"📊 格式化结果: {formatted_result}")
                        
                        # 发送检测结果
                        detection_message = send_detection_result(
                            current_waypoint.name, detections, detection_manager
                        )
                    else:
                        print(f"❌ 航点 {current_waypoint.name} 未检测到物体")
                        # 发送空检测结果
                        empty_message = {
                            'position': current_waypoint.name,
                            'detections': '{e0m0p0w0t0}',
                            'total_objects': 0,
                            'timestamp': datetime.now().isoformat()
                        }
                        print(f"📡 发送空检测结果: {empty_message}")
                    
                    # 更新检测数量
                    flight_manager.update_detection_count(len(detections))
                    
                    if detections and current_waypoint.approach_objects:
                        # 如果检测到物体且需要逼近，则进行逼近操作
                        await approach_detected_objects(
                            drone, 
                            vision_system, 
                            detection_manager, 
                            current_waypoint.name, 
                            waypoint_position
                        )
                    else:
                        if not detections:
                            print(f"❌ 航点 {current_waypoint.name} 未检测到物体")
                        else:
                            print(f"ℹ️  航点 {current_waypoint.name} 检测到物体但跳过逼近")
                    
                    # 退出检测模式
                    flight_manager.exit_special_mode()
            
            # 完成当前航点
            flight_manager.mark_waypoint_completed()
            
            # 显示进度 (飞行管理器 + 航点处理管理器)
            progress = flight_manager.get_progress_info()
            waypoint_stats = WAYPOINT_MANAGER.get_processing_stats()
            print(f"飞行进度: {progress['progress_percentage']:.1f}% ({progress['completed_waypoints']}/{progress['total_waypoints']})")
            print(f"处理进度: {waypoint_stats['completion_percentage']:.1f}% ({waypoint_stats['processed_count']}/{waypoint_stats['total_waypoints']})")
        
        print("🏁 所有航点飞行完成")
        # ==================== 任务总结 ====================
        flight_manager.end_mission()
        
        # 显示优化航点管理器统计
        waypoint_stats = WAYPOINT_MANAGER.get_processing_stats()
        print(f"\n📊 航点处理统计 (优化系统):")
        print(f"   总航点数: {waypoint_stats['total_waypoints']}")
        print(f"   已处理数: {waypoint_stats['processed_count']}")
        print(f"   完成率: {waypoint_stats['completion_percentage']:.1f}%")
        print(f"   剩余航点: {waypoint_stats['remaining_count']}")
        
        # 显示检测统计
        detection_summary = detection_manager.get_detection_summary()
        print(f"\n📈 检测统计:")
        print(f"   总航点: {detection_summary['total_waypoints']}")
        print(f"   总检测: {detection_summary['total_detections']}")
        print(f"   检测率: {detection_summary['detection_rate']:.1%}")
        if detection_summary['most_common_class']:
            print(f"   最常见物体: {detection_summary['most_common_class'][0]} ({detection_summary['most_common_class'][1]}次)")
        # 导出日志
        try:
            flight_manager.export_flight_log("flight_mission_log.txt")
            detection_manager.export_detection_log("detection_mission_log.txt")
            print("📄 任务日志已导出")
        except Exception as e:
            print(f"⚠️  导出日志时出错: {e}")
    except Exception as e:
        print(f"❌ 飞行过程中发生异常: {e}")
        
    finally:
        # ==================== 安全降落和清理 ====================
        print("🛬 执行安全降落和清理...")
        
        try:
            # 高度低于0.5米时kill, 平时少摔一些
            async for pos_vel_ned in drone.telemetry.position_velocity_ned():
                if -pos_vel_ned.position.down_m < 0.05:
                    await drone.action.kill()
                    break
        except Exception as e:
            print(f"降落过程中出现错误: {e}")
        
        try:
            # 停止offboard模式
            print("-- Stopping offboard")
            await drone.offboard.stop()
        except Exception as e:
            print(f"停止offboard模式失败: {e}")
        
        try:
            # 清理资源
            vision_system.cleanup()
            detection_manager.cleanup()
            # 重置航点管理器状态 (如果需要重新开始任务)
            # WAYPOINT_MANAGER.reset_processing_state()  # 取消注释以重置状态
            print("✅ 资源清理完成")
        except Exception as e:
            print(f"资源清理失败: {e}")
        
        print("🎉 任务完成！")

        if 'ser_port' in locals():
            ser_port.close()

if __name__ == "__main__":
    mylist = []
    asyncio.run(run())