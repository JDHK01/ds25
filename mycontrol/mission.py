import control as ctrl
from mavsdk.offboard import VelocityBodyYawspeed
import asyncio
#在主程序中调用的飞行任务函数
#3.37 2.19
async def main_mission(drone):
    """
    执行飞行任务
    :param drone: mavsdk.System对象
    """
    await ctrl.goto_position_ned(drone, 0.0, 0.0, -1.3, 0.0, 10)
    # await ctrl.goto_position_ned(drone, 3.41, 0.0, -1.3, 0.0,10)
    # await ctrl.goto_position_ned(drone, 3.41, -2.23, -1.3, 0.0,10)
    await ctrl.goto_position_ned(drone, 0.0, 0.0, -0.5, 0.0, 5)

    # await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.3, 0.0, 0.0, 0.0))
    # await asyncio.sleep(3)
    # await drone.offboard.set_velocity_body(VelocityBodyYawspeed(-0.3, 0.0, 0.0, 0.0))
    # await asyncio.sleep(3)
    # await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.1, 0.0, 0.0))
    # await asyncio.sleep(8)
    # await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    # 定点测量
    # await ctrl.goto_position_ned(drone, 3.37, 0.0, -0.5, 0.0,10)
    # await ctrl.goto_position_ned(drone, 3.37, -2.19, -0.5, 0.0,10)
    # await ctrl.goto_position_ned(drone, 3.37, -2.19, 0.0, 0.0,10)