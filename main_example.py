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
sys.path.append("/home/by/ds25/temp/lib")
sys.path.append("/home/by/ds25/temp/gc")
from ser import * 
import plan_pro_max
# from flightpath import *
from mycontrol import drone_ctrl as ctrl
# from mono_camera import *
# from detect_manager import *

import Jetson.GPIO as GPIO

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

# 边飞边检测

# 运行
async def run():

    # ====================接收串口屏的消息=======================
    #记得改sys.path.append("/home/by/ds25/temp/lib")，串口的路径
    DRONERECEIVE = '#'
    DRONESEND = '$ANI'
    LORA_PACKET_FOOTER = "%"
    # --- 串口设置 (自动探测) ---
    BAUD_RATE = 9600

    # 创建串口对象，不指定port让其自动探测
    ser_port = SerialPort(port=None, baudrate=BAUD_RATE)
    print("正在探测并连接串口...")
    if not ser_port.open():
        print("串口连接失败，程序退出")
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
        await asyncio.sleep(0.01)
    print('收到禁飞区:')
    print(mylist)
    routine = plan_pro_max.get_mapping_result(tuple(sorted(mylist)))
    print('使用的航点清单:')
    print(routine)

    """边飞行边检测的主函数"""
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

    drone_ctrl = ctrl.Drone_Controller(routine)#飞行控制实例化
    await drone_ctrl.goto_position_ned(drone, 0.0, 0.0, -2.0, 0.0, 0)
    async for pos in drone.telemetry.position_velocity_ned():
        print(f"[起飞] 已起飞到高度 {-pos.position.down_m:.2f} 米")
        await drone_ctrl.print_current_position(drone)
        if -pos.position.down_m > 1.2:
            break
    await drone_ctrl.print_current_position(drone)
    await drone_ctrl.goto_position_ned(drone, 0.0, 0.0, -1.20, 0.0, 3)
    await drone_ctrl.pilot_plan(drone, ser_port)

    # await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.3, 0.0))
    # async for pos in drone.telemetry.position_velocity_ned():
    #     if -pos.position.down_m < 0.11:
    #         await drone.action.kill()
    #         break
    
    print("🎉 任务完成！")

if __name__ == "__main__":
    # Run the asyncio loop
    mylist = []
    asyncio.run(run())
