import asyncio
from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed

#位置控制
async def goto_position_ned(drone, north, east, down, yaw,duiration):
    """
    控制无人机以指定位置进行offboard控制,并等待到达(欧氏距离小于0.1)后再等待3秒。
    :param drone: mavsdk.System对象
    :param north, east, down, yaw: 目标NED位置和偏航角
    """
    print(f"前往目标位置: N={north:.2f}, E={east:.2f}, D={down:.2f}, Yaw={yaw:.2f}")
    tf_n=-east
    tf_e=north
    await drone.offboard.set_position_ned(PositionNedYaw(tf_n, tf_e, down, yaw+90.0))
    await asyncio.sleep(duiration)
    print("到达目标位置")

def mytf((in_pos_n,in_pos_e,in_pos_d, in_yaw)):
    #in 是从无人机读取的ned，out是转换后的ned（前右下）
    out_pos_n = in_pos_e
    out_pos_e = -in_pos_n
    out_pos_d = in_pos_d
    out_yaw = in_yaw - 90.0
    return (out_pos_n, out_pos_e, out_pos_d, out_yaw)
