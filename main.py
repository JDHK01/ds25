#!/usr/bin/env python3

"""
边飞行边检测的无人机控制系统
- 按照预定义路径飞行
- 实时进行目标检测
- 检测到目标时暂停飞行，执行视觉导航
- 视觉导航完成后恢复飞行路径
"""



import sys
sys.path.append("/home/by/wrj/mycontrol")
sys.path.append("/home/by/wrj/vision/cv")
sys.path.append("/home/by/wrj/lib")
from flightpath import *
from control import *
from mono_camera import *
from detect_manager import *
from generate_system import *
from ser import SerialPort

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

# async def get_current_position(drone) -> Tuple[float, float, float, float]:
#     """获取当前位置"""
    # async for pos_vel_ned in drone.telemetry.position_velocity_ned():
    #     return (
    #         pos_vel_ned.position.north_m,
    #         pos_vel_ned.position.east_m, 
    #         pos_vel_ned.position.down_m,
    #         0.0  # yaw暂时设为0
    #     )


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
    

# 读取当前位置
async def get_current_position(drone) -> Tuple[float, float, float, float]:
    """获取当前位置和yaw角度"""
    # 先获取yaw角度
    async for attitude in drone.telemetry.attitude_euler():
        yaw_deg = attitude.yaw_deg
        break
    else:
        yaw_deg = 0.0
        
    # 再获取位置
    async for pos_vel_ned in drone.telemetry.position_velocity_ned():
        # 加入我自己的坐标转换逻辑
        return mytf(
            pos_vel_ned.position.north_m,
            pos_vel_ned.position.east_m,
            pos_vel_ned.position.down_m,
            yaw_deg
        )
    

COORDINATES = generate_coordinate_system()
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
    根据用户提供的航点名称列表创建飞行计划
    
    Args:
        waypoint_names: 航点名称列表，如 ["A1B1", "A2B2"]
        height: 飞行高度
        duration: 在每个航点的停留时间
    Returns:
        配置好的FlightPathManager
    """
    
    # 创建飞行路径管理器
    flight_manager = FlightPathManager()
    flight_manager.create_waypoints_from_user_format(
        waypoint_names=waypoint_names,
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
async def run(user_waypoint_list: List[str] = None, enable_serial_listening: bool = False):
    """边飞行边检测的主函数"""
    waypoint_listener = None
    
    if enable_serial_listening:
        waypoint_listener = WaypointListener()
        if waypoint_listener.start_listening():
            print("等待串口航点消息...")
            user_waypoint_list = waypoint_listener.get_waypoints(timeout=30)
            if user_waypoint_list:
                print(f"收到航点列表: {user_waypoint_list}")
            else:
                print("未收到航点消息，使用默认航点")
                user_waypoint_list = ["A9B1", "A8B1", "A7B1", "A3B3"]
        else:
            print("串口监听启动失败，使用默认航点")
            user_waypoint_list = ["A9B1", "A8B1", "A7B1", "A3B3"]
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

    # ==================== 配置系统 ====================
    # 相机配置
    camera_config = CameraConfig(
        width=640,
        height=480,
        fps=30,
        device_id=0,
        offset_forward=0.0,
        offset_right=0.0,
        offset_down=0.05,
        show_window=False
    )
    
    # 导航配置
    navigation_config = {
        'position_tolerance': 100,
        'min_target_area': 1000,
        'max_velocity': 0.5,
        'offset_compensation_gain': 0.6,
        'alignment_duration': 0.5,
        'completion_tolerance': 80
    }
    
    # PID配置
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
    
    # ==================== 创建检测管理器 ====================
    detection_manager = DetectionManager(
        model_path="vision/yolo/best.pt",
        conf_threshold=0.5,
        device="cuda",  # 或者 "cuda" 如果有GPU
        camera_id=0
    )
    
    # ==================== 定义飞行路径 ====================
    flight_manager = create_waypoint_flight_plan(user_waypoint_list, HEIGHT, DURATION)
    
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
            # 飞往航点
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
            
            # ==================== 航点检测和逼近 ====================
            if current_waypoint.enable_detection:
                print(f"🔍 在航点 {current_waypoint.name} 开始物体检测")
                
                # 进入检测模式
                flight_manager.enter_detection_mode()
                
                # 从摄像头检测物体
                waypoint_position = (current_waypoint.north, current_waypoint.east, current_waypoint.down)
                detections = detection_manager.detect_objects_from_camera(
                    current_waypoint.name, 
                    waypoint_position
                )
                
                # 获取检测统计和格式化结果
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
                
                # 保存检测信息
                detection_manager.save_detection_info(
                    current_waypoint.name, 
                    waypoint_position, 
                    detections
                )
                
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
            
            # 显示进度
            progress = flight_manager.get_progress_info()
            print(f"📊 任务进度: {progress['progress_percentage']:.1f}% ({progress['completed_waypoints']}/{progress['total_waypoints']})")
        
        print("🏁 所有航点飞行完成")
        # ==================== 任务总结 ====================
        flight_manager.end_mission()
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
            if waypoint_listener:
                waypoint_listener.stop_listening()
            print("✅ 资源清理完成")
        except Exception as e:
            print(f"资源清理失败: {e}")
        
        print("🎉 任务完成！")

# 主程序入口
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

if __name__ == "__main__":
    # 模拟传入过程
    try:
        # 创建串口对象
        ser_port = SerialPort(port=SERIAL_PORT, baudrate=BAUD_RATE)
        if not ser_port.open():
            print("无法打开")

        ser_port.register_packet_handler(REICEIVE, command_handler)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 确保串口被关闭
        if 'ser_port' in locals():
            ser_port.close()
            
    # user_waypoints = ["A9B1", "A8B1", "A7B1", "A3B3"]
    
    # 启用串口监听模式，注释掉下面一行则使用默认航点
    # asyncio.run(run(user_waypoints, enable_serial_listening=True))
    
    # 默认模式（使用预设航点）
    # asyncio.run(run(user_waypoints))