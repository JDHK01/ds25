import asyncio
from mavsdk.offboard import PositionNedYaw

#位置控制
async def goto_position_ned(drone, north, east, down, yaw, duiration):
    """
    :param drone: mavsdk.System对象
    :param north, east, down, yaw: 目标NED位置和偏航角
    """
    print(f"前往目标位置: N={north:.2f}, E={east:.2f}, D={down:.2f}, Yaw={yaw:.2f}")
    tf_n=-east
    tf_e=north
    await drone.offboard.set_position_ned(PositionNedYaw(tf_n, tf_e, down, yaw+90.0))
    await asyncio.sleep(duiration)
    print("到达目标位置")