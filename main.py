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

async def fly_to_waypoint_with_detection(drone, waypoint: Waypoint, 
                                       detection_manager: DetectionManager,
                                       flight_manager: FlightPathManager) -> bool:
    """飞向航点并同时进行目标检测，返回是否检测到目标"""
    print(f"飞向航点: {waypoint.name} ({waypoint.north:.1f}, {waypoint.east:.1f}, {waypoint.down:.1f})")
    # 设置目标位置
    # await drone.offboard.set_position_ned(
    #     PositionNedYaw(waypoint.north, waypoint.east, waypoint.down, waypoint.yaw)
    # )
    await ctrl.goto_position_ned(drone, waypoint.north, waypoint.east, waypoint.down, waypoint.yaw, 10)
    start_time = time.time()
    # 感觉这里就相当于time.sleep(), 不太确定
    while time.time() - start_time < waypoint.duration:
        # 检查是否检测到目标
        if await detection_manager.check_for_targets():
            # 获取当前位置并暂停飞行
            current_pos = await get_current_position(drone)
            flight_manager.pause_for_vision_navigation(current_pos)
            return True
            
        await asyncio.sleep(0.05)  # 100Hz检测频率
    
    print(f"到达航点: {waypoint.name}")
    return False

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
        device_id=7,
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
        'offset_compensation_gain': 0.3,
        'alignment_duration': 1.0,
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
        target_mode=TargetMode.FRONT,
        navigation_config=navigation_config,
        pid_config=pid_config
    )
    
    # 创建管理器
    detection_manager = DetectionManager(vision_system)
    
    # ==================== 定义飞行路径 ====================
    flight_waypoints = [
        Waypoint(0.0, 0.0, -1.3, 0.0, 8.0, "起飞点"),
        Waypoint(2.0, 0.0, -1.3, 0.0, 8.0, "前进2米"),
        Waypoint(2.0, 2.0, -1.3, 0.0, 8.0, "右转2米"),
        Waypoint(0.0, 2.0, -1.3, 0.0, 8.0, "后退2米"),
        Waypoint(0.0, 0.0, -1.3, 0.0, 8.0, "回到原点"),
        Waypoint(0.0, 0.0, -0.5, 0.0, 5.0, "降低高度"),
        Waypoint(0.0, 0.0, 0.0, 0.0, 5.0, "准备降落")
    ]
    
    flight_manager = FlightPathManager(flight_waypoints)
    
    print(f"飞行路径规划完成，共{len(flight_waypoints)}个航点")
    
    # ==================== 主飞行循环 ====================
    while not flight_manager.is_completed():
        current_waypoint = flight_manager.get_current_waypoint()
        if current_waypoint is None:
            break
        print(f"📍 进度: {flight_manager.get_progress()}")
        # 飞向航点并检测目标
        target_detected = await fly_to_waypoint_with_detection(
            drone, current_waypoint, detection_manager, flight_manager
        )
        if target_detected:
            # ========== 目标检测到，开始视觉导航 ==========
            print("🎯 开始视觉导航...")
            detection_manager.disable_detection()  # 暂停检测
            
            # 执行视觉导航
            await drone_control_loop(vision_system, drone)
            
            print("✅ 视觉导航完成，恢复飞行路径")
            
            # 恢复到暂停位置
            if flight_manager.paused_position:
                print(f"🔄 返回暂停位置: {flight_manager.paused_position}")
                await goto_position_ned(
                    drone, 
                    flight_manager.paused_position[0],
                    flight_manager.paused_position[1], 
                    flight_manager.paused_position[2],
                    flight_manager.paused_position[3], 
                    5.0
                )
            
            # 恢复状态
            flight_manager.resume_flight_path()
            detection_manager.enable_detection()
            vision_system.reset_task()  # 重置视觉系统
            
            # 继续当前航点（因为之前被中断了）
            continue
        else:
            # 正常到达航点，移动到下一个
            flight_manager.next_waypoint()
    
    print("🏁 所有航点飞行完成")
    
    # ==================== 降落 ====================
    print("🛬 开始降落...")
    
    # 高度低于0.2米时kill
    async for pos_vel_ned in drone.telemetry.position_velocity_ned():
        if -pos_vel_ned.position.down_m < 0.2:
            await drone.action.kill()
            break
    
    # 停止offboard模式
    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except Exception as error:
        print(f"Stopping offboard mode failed with error: {error}")
    
    # 清理资源
    vision_system.cleanup()
    print("🎉 任务完成！")

if __name__ == "__main__":
    
    
    # Run the asyncio loop
    asyncio.run(run())
