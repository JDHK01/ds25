import asyncio
import math
from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed
import Jetson.GPIO as GPIO

import asyncio

from mavsdk.offboard import ( PositionNedYaw, VelocityBodyYawspeed)

class Drone_Controller:
    def __init__(self, path_label=["A9B1"]):
        """
        初始化函数,将路径标签转换为NED坐标路径点。
        """
        self.path_label = path_label
        self.label_map = {
            "A9B1": (0.0, 0.0),
            "A8B1": (0.5, 0.0),
            "A7B1": (1.0, 0.0),
            "A6B1": (1.5, 0.0),
            "A5B1": (2.0, 0.0),
            "A4B1": (2.5, 0.0),
            "A3B1": (3.0, 0.0),
            "A2B1": (3.5, 0.0),
            "A1B1": (4.0, 0.0),
            "A9B2": (0.0, 0.5),
            "A8B2": (0.5, 0.5),
            "A7B2": (1.0, 0.5),
            "A6B2": (1.5, 0.5),
            "A5B2": (2.0, 0.5),
            "A4B2": (2.5, 0.5),
            "A3B2": (3.0, 0.5),
            "A2B2": (3.5, 0.5),
            "A1B2": (4.0, 0.5),
            "A9B3": (0.0, 1.0),
            "A8B3": (0.5, 1.0),
            "A7B3": (1.0, 1.0),
            "A6B3": (1.5, 1.0),
            "A5B3": (2.0, 1.0),
            "A4B3": (2.5, 1.0),
            "A3B3": (3.0, 1.0),
            "A2B3": (3.5, 1.0),
            "A1B3": (4.0, 1.0),
            "A9B4": (0.0, 1.5),
            "A8B4": (0.5, 1.5),
            "A7B4": (1.0, 1.5),
            "A6B4": (1.5, 1.5),
            "A5B4": (2.0, 1.5),
            "A4B4": (2.5, 1.5),
            "A3B4": (3.0, 1.5),
            "A2B4": (3.5, 1.5),
            "A1B4": (4.0, 1.5),
            "A9B5": (0.0, 2.0),
            "A8B5": (0.5, 2.0),
            "A7B5": (1.0, 2.0),
            "A6B5": (1.5, 2.0),
            "A5B5": (2.0, 2.0),
            "A4B5": (2.5, 2.0),
            "A3B5": (3.0, 2.0),
            "A2B5": (3.5, 2.0),
            "A1B5": (4.0, 2.0),
            "A9B6": (0.0, 2.5),
            "A8B6": (0.5, 2.5),
            "A7B6": (1.0, 2.5),
            "A6B6": (1.5, 2.5),
            "A5B6": (2.0, 2.5),
            "A4B6": (2.5, 2.5),
            "A3B6": (3.0, 2.5),
            "A2B6": (3.5, 2.5),
            "A1B6": (4.0, 2.5),
            "A9B7": (0.0, 3.0),
            "A8B7": (0.5, 3.0),
            "A7B7": (1.0, 3.0),
            "A6B7": (1.5, 3.0),
            "A5B7": (2.0, 3.0),
            "A4B7": (2.5, 3.0),
            "A3B7": (3.0, 3.0),
            "A2B7": (3.5, 3.0),
            "A1B7": (4.0, 3.0),
        }
        self.path = self.convert_path(path_label)

    def convert_path(self, path_label):
        path = []
        for label in path_label:
            if label in self.label_map:
                path.append(self.label_map[label])
            else:
                print(f"未知路径标签: {label}")
        return path

    @staticmethod
    def mytf(in_pos_n, in_pos_e, in_pos_d, in_yaw):
        # 将无人机NED坐标转换为实际前右下系统（前为N）
        out_pos_n = in_pos_e
        out_pos_e = -in_pos_n
        out_pos_d = in_pos_d
        out_yaw = in_yaw - 90.0
        return (out_pos_n, out_pos_e, out_pos_d, out_yaw)

    async def goto_position_ned(self, drone, north, east, down, yaw, duration):
        print(f"[位置控制] 前往 N={north:.2f}, E={east:.2f}, D={down:.2f}, Yaw={yaw:.2f}")
        tf_n = -east
        tf_e = north
        await drone.offboard.set_position_ned(PositionNedYaw(tf_n, tf_e, down, yaw + 90.0))
        await asyncio.sleep(duration)
        print("到达目标位置")

    @staticmethod
    def dst(current_point, next_point):
        return math.hypot(current_point[0] - next_point[0], current_point[1] - next_point[1])
    
    @staticmethod
    def dst3d(current_point, next_point):
        return math.sqrt((current_point[0] - next_point[0]) ** 2 + (current_point[1] - next_point[1]) ** 2 + (current_point[2] - next_point[2]) ** 2)

    async def goto_next(self, drone, current_point, next_point):
        dx = next_point[0] - current_point[0]
        dy = next_point[1] - current_point[1]
        distance = self.dst(current_point, next_point)
        if distance == 0:
            return

        direction = (dx / distance, dy / distance)
        k=0
        if dx ==0:
            if distance == 0.5:
                k = 0.3
                duration = 2
            else:
                k = 0.5
                duration = distance / k +0.25
        else:
            if distance == 0.5:
                k = 0.3
                duration = 1.7
            else:
                k = 0.5
                duration = distance / k
        vx = direction[0] * k
        vy = direction[1] * k

        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, 0, 0))
        await asyncio.sleep(duration)
        await self.goto_position_ned(drone, next_point[0], next_point[1], -1.20, 0.0, 0.0)
        async for pos in drone.telemetry.position_velocity_ned():
            my_pos = self.mytf(pos.position.north_m, pos.position.east_m, pos.position.down_m, 0)
            if self.dst3d(my_pos[:3], (*next_point,-1.2)) < 0.1:
                print(f"[到达] 已到达 {next_point}")
                break
        await self.print_current_position(drone)

    async def print_current_position(self, drone):
        async for pos in drone.telemetry.position_velocity_ned():
            my_pos = self.mytf(pos.position.north_m, pos.position.east_m, pos.position.down_m, 0)
            print(f"[当前位置] N={my_pos[0]:.2f}, E={my_pos[1]:.2f}, D={my_pos[2]:.2f}")
            break  # 只打印一次位置

    async def land_from_A8B1(self, drone):
        print("[降落] A8B1逻辑")
        (x, y) = self.label_map["A8B1"]
        await self.goto_position_ned(drone, x, y, -0.5, 0.0, 5)#duiration 待测
        await self.print_current_position(drone)
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(-0.5, 0, 0.5, 0))  # 45度速度控制
        async for pos in drone.telemetry.position_velocity_ned():
            if -pos.position.down_m < 0.12:
                print("[降落] 已触地")
                await drone.action.kill()
                break

    async def land_from_A9B2(self, drone):
        print("[降落] A9B2逻辑")
        (x, y) = self.label_map["A9B2"]
        await self.goto_position_ned(drone, x, y, -0.5, 0.0, 5)
        await self.print_current_position(drone)
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, -0.5, 0.5, 0))  # 45度速度控制
        async for pos in drone.telemetry.position_velocity_ned():
            if -pos.position.down_m < 0.12:
                print("[降落] 已触地")
                await drone.action.kill()
                break

    async def pilot_plan(self, drone):
        print("[任务] 启动飞行任务")
        if not self.path:
            print("路径为空，无法执行")
            return

        current_point = self.path[0]
        for next_point in self.path[1:]:
            if current_point == self.label_map["A8B1"]  and next_point == self.label_map["A9B1"]:
                await self.land_from_A8B1(drone)
                return
            elif current_point == self.label_map["A9B2"] and next_point == self.label_map["A9B1"]:
                await self.land_from_A9B2(drone)
                return
            await self.goto_next(drone, current_point, next_point)
            current_point = next_point