#!/usr/bin/env python3
"""
飞行路径管理模块
提供航点数据结构和飞行路径管理功能
"""

import asyncio
import time
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class WaypointStatus(Enum):
    """航点状态"""
    PENDING = "pending"          # 待飞行
    IN_PROGRESS = "in_progress"  # 飞行中
    ARRIVED = "arrived"          # 已到达
    DETECTION_MODE = "detection" # 检测模式
    COMPLETED = "completed"      # 已完成

# 描述飞行管理器的工作状态
class FlightMode(Enum):
    """飞行模式"""
    NAVIGATION = "navigation"    # 导航飞行
    DETECTION = "detection"      # 检测模式
    APPROACHING = "approaching"  # 物体逼近模式
    PAUSED = "paused"           # 暂停

@dataclass
class Waypoint:
    """航点数据结构"""
    north: float                              # 北向坐标 (m)
    east: float                               # 东向坐标 (m)
    down: float                               # 下向坐标 (m, 负值表示高度)
    yaw: float                                # 偏航角 (degrees)
    duration: float                           # 在该点停留时间 (seconds)
    name: str                                 # 航点名称
    
    # 状态信息
    status: WaypointStatus = WaypointStatus.PENDING
    arrival_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    detection_count: int = 0                  # 在此航点检测到的物体数量
    
    # 检测配置
    enable_detection: bool = True             # 是否在此航点进行检测
    detection_timeout: float = 2.0            # 检测超时时间 (seconds)
    approach_objects: bool = True             # 是否逼近检测到的物体

    def __str__(self):
        return f"Waypoint({self.name}: N={self.north:.1f}, E={self.east:.1f}, D={self.down:.1f})"

class FlightPathManager:
    """飞行路径管理器"""

    def __init__(self, waypoints: List[Waypoint] = None):
        """
        初始化飞行路径管理器
        
        输入:
            航点列表
        """
        self.waypoints: List[Waypoint] = waypoints or []
        self.current_index: int = 0
        self.flight_mode: FlightMode = FlightMode.NAVIGATION
        self.is_paused: bool = False
        
        # 任务状态
        self.mission_start_time: Optional[datetime] = None
        self.mission_end_time: Optional[datetime] = None
        self.total_detections: int = 0
        
        # 统计信息
        self.flight_distance: float = 0.0
        self.detection_success_rate: float = 0.0
        
        print(f"飞行路径管理器初始化完成 - 航点数量: {len(self.waypoints)}")
    
    def add_waypoint(self, waypoint: Waypoint):
        """添加航点"""
        self.waypoints.append(waypoint)
        print(f"添加航点: {waypoint.name}")
    
    def add_waypoints_from_coordinates(self, coordinates: List[Tuple[float, float]], 
                                     height: float = -1.0, yaw: float = 0.0, 
                                     duration: float = 3.0):
        """
        从坐标列表批量添加航点
        
        Args:
            coordinates: 坐标列表 [(x, y), ...]
            height: 飞行高度 (负值)
            yaw: 偏航角
            duration: 在每个航点的停留时间
        """
        for i, (x, y) in enumerate(coordinates):
            waypoint = Waypoint(
                north=x,
                east=y,
                down=height,
                yaw=yaw,
                duration=duration,
                name=f"WP_{i+1:02d}"
            )
            self.add_waypoint(waypoint)
    
    def create_waypoints_from_user_format(self, waypoint_names: List[str],
                                        coordinate_dict: Dict[str, Tuple[float, float]],
                                        height: float = -1.0, yaw: float = 0.0,
                                        duration: float = 3.0):
        """
        根据用户格式创建航点 (如 ["A1B1", "A2B2"])
        
        Args:
            waypoint_names: 航点名称列表 (如 ["A1B1", "A2B2"])
            coordinate_dict: 坐标字典 (如 {"A1B1": (4.0, 0.5)})
            height: 飞行高度
            yaw: 偏航角  
            duration: 停留时间
        """
        self.waypoints.clear()
        self.current_index = 0
        
        for name in waypoint_names:
            if name in coordinate_dict:
                x, y = coordinate_dict[name]
                waypoint = Waypoint(
                    north=x,
                    east=y, 
                    down=height,
                    yaw=yaw,
                    duration=duration,
                    name=name
                )
                self.add_waypoint(waypoint)
            else:
                print(f"警告: 航点 {name} 未在坐标字典中找到")
    
    def get_current_waypoint(self) -> Optional[Waypoint]:
        """获取当前航点"""
        if 0 <= self.current_index < len(self.waypoints):
            return self.waypoints[self.current_index]
        return None
    
    def get_next_waypoint(self) -> Optional[Waypoint]:
        """获取下一个航点"""
        next_index = self.current_index + 1
        if next_index < len(self.waypoints):
            return self.waypoints[next_index]
        return None
    
    def mark_waypoint_arrived(self):
        """标记当前航点为到达状态"""
        current = self.get_current_waypoint()
        if current:
            current.status = WaypointStatus.ARRIVED
            current.arrival_time = datetime.now()
            print(f"航点 {current.name} 已到达")
    
    def mark_waypoint_completed(self):
        """标记当前航点为完成状态并前进到下一个"""
        current = self.get_current_waypoint()
        if current:
            current.status = WaypointStatus.COMPLETED
            current.completion_time = datetime.now()
            print(f"航点 {current.name} 已完成")
            
        self.next_waypoint()
    
    def next_waypoint(self):
        """前进到下一个航点"""
        if self.current_index < len(self.waypoints) - 1:
            self.current_index += 1
            current = self.get_current_waypoint()
            if current:
                current.status = WaypointStatus.IN_PROGRESS
                print(f"前往下一个航点: {current.name}")
                return True
        else:
            print("所有航点已完成")
            return False
    
    def is_completed(self) -> bool:
        """检查任务是否完成"""
        return self.current_index >= len(self.waypoints)
    
    def pause_mission(self):
        """暂停任务"""
        self.is_paused = True
        self.flight_mode = FlightMode.PAUSED
        print("任务已暂停")
    
    def resume_mission(self):
        """恢复任务"""
        self.is_paused = False
        self.flight_mode = FlightMode.NAVIGATION
        print("任务已恢复")
    
    def enter_detection_mode(self):
        """进入检测模式"""
        self.flight_mode = FlightMode.DETECTION
        current = self.get_current_waypoint()
        if current:
            current.status = WaypointStatus.DETECTION_MODE
            print(f"在航点 {current.name} 进入检测模式")
    
    def enter_approaching_mode(self):
        """进入物体逼近模式"""
        self.flight_mode = FlightMode.APPROACHING
        print("进入物体逼近模式")
    
    def exit_special_mode(self):
        """退出特殊模式，返回导航"""
        self.flight_mode = FlightMode.NAVIGATION
        current = self.get_current_waypoint()
        if current and current.status == WaypointStatus.DETECTION_MODE:
            current.status = WaypointStatus.ARRIVED
        print("返回导航模式")
    
    def update_detection_count(self, count: int):
        """更新当前航点的检测数量"""
        current = self.get_current_waypoint()
        if current:
            current.detection_count = count
            self.total_detections += count
            print(f"航点 {current.name} 检测到 {count} 个物体")
    
    def get_progress_info(self) -> Dict[str, Any]:
        """获取任务进度信息"""
        completed_waypoints = sum(1 for wp in self.waypoints if wp.status == WaypointStatus.COMPLETED)
        progress_percentage = (completed_waypoints / len(self.waypoints)) * 100 if self.waypoints else 0
        
        current = self.get_current_waypoint()
        current_info = {
            'name': current.name if current else None,
            'status': current.status.value if current else None,
            'position': (current.north, current.east, current.down) if current else None
        }
        
        return {
            'total_waypoints': len(self.waypoints),
            'completed_waypoints': completed_waypoints,
            'current_index': self.current_index,
            'progress_percentage': progress_percentage,
            'flight_mode': self.flight_mode.value,
            'is_paused': self.is_paused,
            'current_waypoint': current_info,
            'total_detections': self.total_detections
        }
    
    def get_mission_summary(self) -> Dict[str, Any]:
        """获取任务统计摘要"""
        if not self.waypoints:
            return {'error': '无航点数据'}
        
        completed_count = sum(1 for wp in self.waypoints if wp.status == WaypointStatus.COMPLETED)
        detection_count = sum(wp.detection_count for wp in self.waypoints)
        waypoints_with_detections = sum(1 for wp in self.waypoints if wp.detection_count > 0)
        
        detection_success_rate = (waypoints_with_detections / len(self.waypoints)) * 100 if self.waypoints else 0
        
        # 计算任务时间
        mission_duration = None
        if self.mission_start_time:
            end_time = self.mission_end_time or datetime.now()
            mission_duration = (end_time - self.mission_start_time).total_seconds()
        
        return {
            'total_waypoints': len(self.waypoints),
            'completed_waypoints': completed_count,
            'completion_rate': (completed_count / len(self.waypoints)) * 100,
            'total_detections': detection_count,
            'waypoints_with_detections': waypoints_with_detections,
            'detection_success_rate': detection_success_rate,
            'mission_duration': mission_duration,
            'flight_mode': self.flight_mode.value
        }
    
    def start_mission(self):
        """开始任务"""
        self.mission_start_time = datetime.now()
        if self.waypoints:
            self.waypoints[0].status = WaypointStatus.IN_PROGRESS
        print("任务开始")
    
    def end_mission(self):
        """结束任务"""
        self.mission_end_time = datetime.now()
        print("任务结束")
        
        # 打印任务摘要
        summary = self.get_mission_summary()
        print("\n任务摘要:")
        print(f"总航点: {summary['total_waypoints']}")
        print(f"完成航点: {summary['completed_waypoints']}")
        print(f"完成率: {summary['completion_rate']:.1f}%")
        print(f"总检测: {summary['total_detections']}")
        print(f"检测成功率: {summary['detection_success_rate']:.1f}%")
        if summary['mission_duration']:
            print(f"任务时长: {summary['mission_duration']:.1f}秒")
    
    def export_flight_log(self, filename: str = "flight_log.txt"):
        """导出飞行日志"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("无人机飞行任务日志\n")
                f.write("=" * 50 + "\n")
                f.write(f"任务时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                summary = self.get_mission_summary()
                f.write(f"总航点数: {summary['total_waypoints']}\n")
                f.write(f"完成航点数: {summary['completed_waypoints']}\n")
                f.write(f"完成率: {summary['completion_rate']:.1f}%\n")
                f.write(f"总检测数: {summary['total_detections']}\n")
                f.write(f"检测成功率: {summary['detection_success_rate']:.1f}%\n\n")
                
                f.write("航点详情:\n")
                f.write("-" * 30 + "\n")
                
                for i, waypoint in enumerate(self.waypoints, 1):
                    f.write(f"{i}. {waypoint.name}\n")
                    f.write(f"   坐标: N={waypoint.north:.2f}, E={waypoint.east:.2f}, D={waypoint.down:.2f}\n")
                    f.write(f"   状态: {waypoint.status.value}\n")
                    f.write(f"   检测数量: {waypoint.detection_count}\n")
                    if waypoint.arrival_time:
                        f.write(f"   到达时间: {waypoint.arrival_time.strftime('%H:%M:%S')}\n")
                    if waypoint.completion_time:
                        f.write(f"   完成时间: {waypoint.completion_time.strftime('%H:%M:%S')}\n")
                    f.write("\n")
                
            print(f"飞行日志已导出到: {filename}")
            
        except Exception as e:
            print(f"导出飞行日志失败: {e}")

# 测试代码
if __name__ == "__main__":
    # 测试航点管理功能
    print("测试飞行路径管理器...")
    
    # 创建测试航点
    waypoints = [
        Waypoint(0.0, 0.0, -1.0, 0.0, 3.0, "起点"),
        Waypoint(1.0, 0.0, -1.0, 0.0, 5.0, "A1B1"),
        Waypoint(1.0, 1.0, -1.0, 0.0, 5.0, "A1B2"),
        Waypoint(0.0, 1.0, -1.0, 0.0, 5.0, "A2B2"),
        Waypoint(0.0, 0.0, 0.0, 0.0, 3.0, "降落点")
    ]
    
    # 创建管理器
    manager = FlightPathManager(waypoints)
    
    # 开始任务
    manager.start_mission()
    
    # 模拟飞行过程
    while not manager.is_completed():
        current = manager.get_current_waypoint()
        if current:
            print(f"\n当前航点: {current}")
            
            # 模拟到达航点
            time.sleep(1)
            manager.mark_waypoint_arrived()
            
            # 模拟检测过程
            if current.enable_detection:
                manager.enter_detection_mode()
                time.sleep(2)  # 模拟检测时间
                
                # 随机模拟检测结果
                import random
                detection_count = random.randint(0, 3)
                manager.update_detection_count(detection_count)
                
                if detection_count > 0:
                    print(f"检测到 {detection_count} 个物体，开始逼近...")
                    manager.enter_approaching_mode()
                    time.sleep(2)  # 模拟逼近时间
                
                manager.exit_special_mode()
            
            # 完成当前航点
            manager.mark_waypoint_completed()
            
            # 显示进度
            progress = manager.get_progress_info()
            print(f"任务进度: {progress['progress_percentage']:.1f}%")
    
    # 结束任务
    manager.end_mission()
    
    # 导出日志
    manager.export_flight_log("test_flight_log.txt")
    
    print("\n测试完成!")