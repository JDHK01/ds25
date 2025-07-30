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
from flightpath import *
from control import *
from mono_camera import *
from detect_manager import *

import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
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

HEIGHT = -1.1
# 所有的点位置存储, 希望claude能进行优化
A9B1 = (0.0, 0.0)
A8B1 = (0.5, 0.0)
A7B1 = (1.0, 0.0)
A6B1 = (1.5, 0.0)
A5B1 = (2.0, 0.0)
A4B1 = (2.5, 0.0)
A3B1 = (3.0, 0.0)
A2B1 = (3.5, 0.0)
A1B1 = (4.0, 0.0)
A9B2 = (0.0, 0.5)
A8B2 = (0.5, 0.5)
A7B2 = (1.0, 0.5)
A6B2 = (1.5, 0.5)
A5B2 = (2.0, 0.5)
A4B2 = (2.5, 0.5)
A3B2 = (3.0, 0.5)
A2B2 = (3.5, 0.5)
A1B2 = (4.0, 0.5)
A9B3 = (0.0, 1.0)
A8B3 = (0.5, 1.0)
A7B3 = (1.0, 1.0)
A6B3 = (1.5, 1.0)
A5B3 = (2.0, 1.0)
A4B3 = (2.5, 1.0)
A3B3 = (3.0, 1.0)
A2B3 = (3.5, 1.0)
A1B3 = (4.0, 1.0)
A9B4 = (0.0, 1.5)
A8B4 = (0.5, 1.5)
A7B4 = (1.0, 1.5)
A6B4 = (1.5, 1.5)
A5B4 = (2.0, 1.5)
A4B4 = (2.5, 1.5)
A3B4 = (3.0, 1.5)
A2B4 = (3.5, 1.5)
A1B4 = (4.0, 1.5)
A9B5 = (0.0, 2.0)
A8B5 = (0.5, 2.0)
A7B5 = (1.0, 2.0)
A6B5 = (1.5, 2.0)
A5B5 = (2.0, 2.0)
A4B5 = (2.5, 2.0)
A3B5 = (3.0, 2.0)
A2B5 = (3.5, 2.0)
A1B5 = (4.0, 2.0)
A9B6 = (0.0, 2.5)
A8B6 = (0.5, 2.5)
A7B6 = (1.0, 2.5)
A6B6 = (1.5, 2.5)
A5B6 = (2.0, 2.5)
A4B6 = (2.5, 2.5)
A3B6 = (3.0, 2.5)
A2B6 = (3.5, 2.5)
A1B6 = (4.0, 2.5)
A9B7 = (0.0, 3.0)
A8B7 = (0.5, 3.0)
A7B7 = (1.0, 3.0)
A6B7 = (1.5, 3.0)
A5B7 = (2.0, 3.0)
A4B7 = (2.5, 3.0)
A3B7 = (3.0, 3.0)
A2B7 = (3.5, 3.0)
A1B7 = (4.0, 3.0)



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

# 运行
async def run():
    """边飞行边检测的主函数"""
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
    
    
    # ==================== 定义飞行路径 ====================
    flight_waypoints = [
        Waypoint(0.0, 0.0, -1.3, 0.0, 10.0, "起飞"),
        Waypoint(0.0, 0.0, -0.3, 0.0, 5.0, "下降到0.3"),
        Waypoint(1.0, 0.0, -0.3, 0.0, 5.0, "向前1米"),
        Waypoint(1.0, 1.0, -0.3, 0.0, 8.0, "向右1米"),
        Waypoint(1.0, 1.0, 0.0, 0.0, 5.0, "降落")
    ]
    
    flight_manager = FlightPathManager(flight_waypoints)
    
    print(f"飞行路径规划完成，共{len(flight_waypoints)}个航点")
    
    try:
        # ==================== 主飞行循环 ====================
        while not flight_manager.is_completed():
            current_waypoint = flight_manager.get_current_waypoint()
            flight_manager.next_waypoint()
        
        print("🏁 所有航点飞行完成")
        
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
            print("✅ 资源清理完成")
        except Exception as e:
            print(f"资源清理失败: {e}")
        
        print("🎉 任务完成！")

if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(run())