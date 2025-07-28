#!/usr/bin/env python3

"""
Caveat when attempting to run the examples in non-gps environments:

`drone.offboard.stop()` will return a `COMMAND_DENIED` result because it
requires a mode switch to HOLD, something that is currently not supported in a
non-gps environment.
"""
'''
        位置 NED 坐标系
        打印NED坐标系下的参数
    async for pvn in drone.telemetry.position_velocity_ned():
        # NED 位置
        north = pvn.position.north_m  # X (NED)
        east = pvn.position.east_m    # Y (NED)  
        down = pvn.position.down_m    # Z (NED)
'''


import control as ctrl
import asyncio
import vision_guide
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw, VelocityBodyYawspeed)
from mavsdk.telemetry import LandedState
import mission

async def run():
    """ Does Offboard control using position NED coordinates. """

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
        print(f"Starting offboard mode failed \
                with error code: {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return

    print("-- takeoff \
            within local coordinate system")
    
    #==========================执行飞行任务===================================
    #         # 相机配置
    # camera_config = vision_guide.CameraConfig(
    #     width=640,
    #     height=480,
    #     fps=30,
    #     device_id=0,  # 根据实际相机调整
    #     # 相机偏移配置（单位：米）
    #     offset_forward=0.0,   # 相机在无人机前方10cm
    #     offset_right=0.0,     # 相机在中心线上
    #     offset_down=0.0      # 相机在无人机下方5cm
    # )
    
    # # 导航配置
    # navigation_config = {
    #     'position_tolerance': 100,    # 像素容差
    #     'min_target_area': 500,     # 最小目标面积
    #     'max_velocity': 0.5,         # 最大速度 m/s
    #     'offset_compensation_gain': 0.3,  # 偏移补偿增益（0-1）
    #     'alignment_duration': 1.0,   # 对准保持时间（秒）
    #     'completion_tolerance': 100   # 完成任务的像素容差
    # }
    
    # # PID配置
    # pid_config = {
    #     'horizontal': {
    #         'kp': 0.3,
    #         'ki': 0.0,
    #         'kd': 0.0,
    #         'output_limit': 0.5
    #     },
    #     'vertical': {
    #         'kp': 0.3,
    #         'ki': 0.0,
    #         'kd': 0.0,
    #         'output_limit': 0.5
    #     },
    #     'forward': {
    #         'kp': 0.3,
    #         'ki': 0.0,
    #         'kd': 0.0,
    #         'output_limit': 0.3
    #     }
    # }
    
    # # 创建视觉导航系统
    # vision_system = vision_guide.VisionGuidanceSystem(
    #     camera_config=camera_config,
    #     target_mode=vision_guide.TargetMode.DOWN,  # 或 TargetMode.FRONT
    #     navigation_config=navigation_config,
    #     pid_config=pid_config
    # )

    await mission.main_mission(drone)
    # await vision_guide.drone_control_loop(vision_system, drone)
    # await asyncio.sleep(2)
   # ==============================着陆========================================
    print("-- Landing")
    # await ctrl.goto_position_ned(drone, 1.0, 1.0, -0.5, 0.0,10)
    # await ctrl.goto_position_ned(drone, 1.0, 1.0, 0.0, 0.0,5)

    # await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    '''原来的降落代码
    await drone.action.land()
    async for state in drone.telemetry.landed_state():
        if state == LandedState.ON_GROUND:
            break
    '''
    #============尝试刹车降落===========
    for pos
    # ==============================停止offboard模式========================================
    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except Exception as error:
        print(f"Stopping offboard mode failed with error: {error}")
    # ==============================解锁========================================
    print("-- Disarming")
    await drone.action.disarm()


if __name__ == "__main__":
    
    
    # Run the asyncio loop
    asyncio.run(run())
