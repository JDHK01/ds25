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