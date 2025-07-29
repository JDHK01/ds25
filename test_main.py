#!/usr/bin/env python3

"""
无人机飞行测试脚本
- 模拟main.py中的无人机飞行和目标检测过程
- 提供详细的测试日志输出
- 不需要真实的硬件连接
"""

import asyncio
import time
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

# 模拟的数据类型
@dataclass
class MockWaypoint:
    north: float
    east: float  
    down: float
    yaw: float
    duration: float
    name: str

@dataclass 
class MockPosition:
    north_m: float
    east_m: float
    down_m: float

@dataclass
class MockAttitude:
    yaw_deg: float

# 模拟无人机系统
class MockDrone:
    def __init__(self):
        self.position = MockPosition(0.0, 0.0, 0.0)
        self.attitude = MockAttitude(0.0)
        self.is_connected = False
        self.is_armed = False
        self.offboard_active = False
        self.target_position = None
        
    async def connect(self, system_address):
        print(f"🔗 连接到模拟无人机: {system_address}")
        await asyncio.sleep(1)
        self.is_connected = True
        print("✅ 连接成功!")
        
    async def check_connection(self):
        print("🔍 检查连接状态...")
        await asyncio.sleep(0.5)
        return self.is_connected
        
    async def check_health(self):
        print("🏥 检查无人机健康状态...")
        await asyncio.sleep(1)
        print("✅ 本地位置估计OK，无人机可解锁")
        return True
        
    async def arm(self):
        print("🔓 解锁无人机...")
        await asyncio.sleep(1)
        self.is_armed = True
        print("✅ 无人机已解锁")
        
    async def start_offboard(self):
        print("🎮 启动offboard模式...")
        await asyncio.sleep(1)
        self.offboard_active = True
        print("✅ Offboard模式已启动")
        
    async def set_position_ned(self, north, east, down, yaw):
        print(f"📍 设置目标位置: N={north:.1f}, E={east:.1f}, D={down:.1f}, Y={yaw:.1f}°")
        self.target_position = (north, east, down, yaw)
        
    async def goto_position(self, north, east, down, yaw, timeout=10):
        print(f"🛫 飞向目标位置: N={north:.1f}, E={east:.1f}, D={down:.1f}, Y={yaw:.1f}° (超时:{timeout}s)")
        
        # 模拟飞行过程
        start_pos = (self.position.north_m, self.position.east_m, self.position.down_m)
        target_pos = (north, east, down)
        
        steps = 20  # 分20步到达
        for i in range(steps + 1):
            progress = i / steps
            current_north = start_pos[0] + (target_pos[0] - start_pos[0]) * progress
            current_east = start_pos[1] + (target_pos[1] - start_pos[1]) * progress  
            current_down = start_pos[2] + (target_pos[2] - start_pos[2]) * progress
            
            self.position.north_m = current_north
            self.position.east_m = current_east
            self.position.down_m = current_down
            self.attitude.yaw_deg = yaw
            
            if i % 5 == 0:  # 每5步输出一次位置
                print(f"  📊 当前位置: N={current_north:.1f}, E={current_east:.1f}, D={current_down:.1f} ({progress*100:.0f}%)")
                
            await asyncio.sleep(0.1)
            
        print(f"✅ 已到达目标位置: N={north:.1f}, E={east:.1f}, D={down:.1f}")
        
    async def get_current_position(self):
        # 添加一些随机噪声模拟真实传感器
        noise = 0.05
        return (
            self.position.north_m + random.uniform(-noise, noise),
            self.position.east_m + random.uniform(-noise, noise), 
            self.position.down_m + random.uniform(-noise, noise),
            self.attitude.yaw_deg + random.uniform(-1, 1)
        )
        
    async def land(self):
        print("🛬 开始降落...")
        current_down = self.position.down_m
        steps = 10
        for i in range(steps + 1):
            progress = i / steps
            new_down = current_down * (1 - progress)  # 逐渐降到0
            self.position.down_m = new_down
            
            if i % 3 == 0:
                print(f"  📊 降落高度: {-new_down:.1f}m ({progress*100:.0f}%)")
                
            await asyncio.sleep(0.2)
            
        print("✅ 降落完成")
        
    async def stop_offboard(self):
        print("🛑 停止offboard模式...")
        await asyncio.sleep(0.5)
        self.offboard_active = False
        print("✅ Offboard模式已停止")

# 模拟目标检测系统
class MockDetectionManager:
    def __init__(self):
        self.detection_enabled = True
        self.detection_probability = 0.15  # 15%概率检测到目标
        
    async def check_for_targets(self):
        if not self.detection_enabled:
            return False
            
        # 模拟目标检测
        detected = random.random() < self.detection_probability
        if detected:
            print("🎯 检测到目标!")
            return True
        return False
        
    def enable_detection(self):
        print("👁️ 启用目标检测")
        self.detection_enabled = True
        
    def disable_detection(self):
        print("👁️ 禁用目标检测") 
        self.detection_enabled = False

# 模拟飞行路径管理器
class MockFlightPathManager:
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.current_index = 0
        self.paused_position = None
        self.is_paused = False
        
    def get_current_waypoint(self):
        if self.current_index < len(self.waypoints):
            return self.waypoints[self.current_index]
        return None
        
    def next_waypoint(self):
        self.current_index += 1
        print(f"➡️ 移动到下一个航点 (索引: {self.current_index})")
        
    def is_completed(self):
        return self.current_index >= len(self.waypoints)
        
    def pause_for_vision_navigation(self, position):
        print(f"⏸️ 在位置暂停飞行路径: N={position[0]:.1f}, E={position[1]:.1f}, D={position[2]:.1f}")
        self.paused_position = position
        self.is_paused = True
        
    def resume_flight_path(self):
        print("▶️ 恢复飞行路径")
        self.is_paused = False
        
    def get_progress(self):
        if len(self.waypoints) == 0:
            return "0/0"
        return f"{self.current_index + 1}/{len(self.waypoints)}"

# 模拟视觉导航系统
class MockVisionSystem:
    def __init__(self):
        self.task_completed = False
        
    async def execute_vision_navigation(self, drone):
        print("🎯 开始执行视觉导航任务...")
        
        # 模拟视觉导航的不同阶段
        stages = [
            "🔍 搜索目标中...",
            "🎯 锁定目标...", 
            "📐 计算接近路径...",
            "🛫 接近目标...",
            "🎯 精确对准...",
            "✅ 到达目标位置"
        ]
        
        for i, stage in enumerate(stages):
            print(f"  {stage} ({i+1}/{len(stages)})")
            await asyncio.sleep(1 + random.uniform(0, 1))  # 模拟不同阶段用时
            
        self.task_completed = True
        print("🎉 视觉导航任务完成!")
        
    def reset_task(self):
        print("🔄 重置视觉导航系统")
        self.task_completed = False
        
    def cleanup(self):
        print("🧹 清理视觉系统资源")

async def fly_to_waypoint_with_detection(drone, waypoint, detection_manager, flight_manager):
    """飞向航点并同时进行目标检测"""
    print(f"\n🚁 开始飞向航点: {waypoint.name}")
    print(f"   目标坐标: N={waypoint.north:.1f}, E={waypoint.east:.1f}, D={waypoint.down:.1f}")
    print(f"   停留时间: {waypoint.duration}秒")
    
    # 飞向目标位置
    await drone.goto_position(waypoint.north, waypoint.east, waypoint.down, waypoint.yaw, 10)
    
    # 在航点停留并检测目标
    print(f"⏱️ 在航点停留 {waypoint.duration} 秒，同时进行目标检测...")
    start_time = time.time()
    detection_count = 0
    
    while time.time() - start_time < waypoint.duration:
        detection_count += 1
        
        # 检查是否检测到目标
        if await detection_manager.check_for_targets():
            # 获取当前位置并暂停飞行
            current_pos = await drone.get_current_position()
            flight_manager.pause_for_vision_navigation(current_pos)
            print(f"🎯 在第 {detection_count} 次检测时发现目标!")
            return True
            
        if detection_count % 20 == 0:  # 每20次检测输出一次状态
            elapsed = time.time() - start_time
            remaining = waypoint.duration - elapsed
            print(f"  🔍 已检测 {detection_count} 次，剩余停留时间: {remaining:.1f}秒")
            
        await asyncio.sleep(0.05)  # 100Hz检测频率
    
    print(f"✅ 完成航点: {waypoint.name} (共检测 {detection_count} 次)")
    return False

async def test_drone_flight():
    """测试无人机飞行的主函数"""
    print("="*60)
    print("🚁 无人机飞行测试开始")
    print("="*60)
    
    # ==================== 无人机初始化 ====================
    print("\n📋 阶段1: 无人机系统初始化")
    print("-" * 40)
    
    drone = MockDrone()
    await drone.connect("udp://127.0.0.1:14540")
    
    print("⏳ 等待无人机连接...")
    await drone.check_connection()
    
    print("⏳ 检查无人机健康状态...")  
    await drone.check_health()
    
    await drone.arm()
    
    print("🎮 设置初始设定点")
    await drone.set_position_ned(0.0, 0.0, 0.0, 90.0)
    
    await drone.start_offboard()
    
    # ==================== 系统配置 ====================
    print("\n📋 阶段2: 系统配置")
    print("-" * 40)
    
    print("📷 配置相机参数:")
    print("   - 分辨率: 640x480")  
    print("   - 帧率: 30fps")
    print("   - 设备ID: 7")
    print("   - 显示窗口: 关闭")
    
    print("🧭 配置导航参数:")
    print("   - 位置容差: 100")
    print("   - 最小目标区域: 1000") 
    print("   - 最大速度: 0.5m/s")
    print("   - 对准持续时间: 1.0s")
    
    print("🎛️ 配置PID参数:")
    print("   - 水平PID: Kp=0.1, Ki=0.0, Kd=0.0")
    print("   - 垂直PID: Kp=0.1, Ki=0.0, Kd=0.0") 
    print("   - 前进PID: Kp=0.1, Ki=0.0, Kd=0.0")
    
    # 创建管理器
    detection_manager = MockDetectionManager()
    vision_system = MockVisionSystem()
    
    # ==================== 定义飞行路径 ====================
    print("\n📋 阶段3: 飞行路径规划")
    print("-" * 40)
    
    flight_waypoints = [
        MockWaypoint(0.0, 0.0, -1.3, 0.0, 8.0, "起飞点"),
        MockWaypoint(2.0, 0.0, -1.3, 0.0, 8.0, "前进2米"),
        MockWaypoint(2.0, 2.0, -1.3, 0.0, 8.0, "右转2米"),
        MockWaypoint(0.0, 2.0, -1.3, 0.0, 8.0, "后退2米"),
        MockWaypoint(0.0, 0.0, -1.3, 0.0, 8.0, "回到原点"),
        MockWaypoint(0.0, 0.0, -0.5, 0.0, 5.0, "降低高度"),
        MockWaypoint(0.0, 0.0, 0.0, 0.0, 5.0, "准备降落")
    ]
    
    flight_manager = MockFlightPathManager(flight_waypoints)
    
    print(f"📋 飞行路径规划完成，共 {len(flight_waypoints)} 个航点:")
    for i, wp in enumerate(flight_waypoints, 1):
        print(f"   {i}. {wp.name}: N={wp.north}, E={wp.east}, D={wp.down} (停留{wp.duration}s)")
    
    # ==================== 主飞行循环 ====================
    print("\n📋 阶段4: 主飞行循环")
    print("-" * 40)
    
    flight_start_time = time.time()
    vision_navigation_count = 0
    
    while not flight_manager.is_completed():
        current_waypoint = flight_manager.get_current_waypoint()
        if current_waypoint is None:
            break
            
        print(f"\n📍 飞行进度: {flight_manager.get_progress()}")
        
        # 飞向航点并检测目标
        target_detected = await fly_to_waypoint_with_detection(
            drone, current_waypoint, detection_manager, flight_manager
        )
        
        if target_detected:
            vision_navigation_count += 1
            print(f"\n🎯 第 {vision_navigation_count} 次视觉导航开始")
            print("-" * 30)
            
            # ========== 目标检测到，开始视觉导航 ==========
            detection_manager.disable_detection()
            
            # 执行视觉导航
            await vision_system.execute_vision_navigation(drone)
            
            print("✅ 视觉导航完成，准备恢复飞行路径")
            
            # 恢复到暂停位置
            if flight_manager.paused_position:
                pos = flight_manager.paused_position
                print(f"🔄 返回暂停位置: N={pos[0]:.1f}, E={pos[1]:.1f}, D={pos[2]:.1f}")
                await drone.goto_position(pos[0], pos[1], pos[2], pos[3])
            
            # 恢复状态
            flight_manager.resume_flight_path()
            detection_manager.enable_detection()
            vision_system.reset_task()
            
            print("▶️ 继续执行当前航点")
            continue
        else:
            # 正常到达航点，移动到下一个
            flight_manager.next_waypoint()
    
    flight_duration = time.time() - flight_start_time
    
    print(f"\n🏁 所有航点飞行完成!")
    print(f"   总飞行时间: {flight_duration:.1f}秒")
    print(f"   视觉导航次数: {vision_navigation_count}次")
    
    # ==================== 降落 ====================
    print("\n📋 阶段5: 降落过程")
    print("-" * 40)
    
    await drone.land()
    
    # 停止offboard模式
    await drone.stop_offboard()
    
    # 清理资源
    vision_system.cleanup()
    
    print("\n" + "="*60)
    print(f"🎉 测试完成! 总用时: {time.time() - flight_start_time:.1f}秒")
    print("📊 测试统计:")
    print(f"   - 完成航点数: {len(flight_waypoints)}")
    print(f"   - 视觉导航次数: {vision_navigation_count}")
    print(f"   - 平均每航点用时: {flight_duration/len(flight_waypoints):.1f}秒")
    print("="*60)

if __name__ == "__main__":
    print("🚁 无人机飞行测试脚本")
    print("⚠️  注意: 这是模拟测试，不需要真实硬件")
    print()
    
    try:
        asyncio.run(test_drone_flight())
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出现错误: {e}")
        import traceback
        traceback.print_exc()