from typing import List, Optional, Tuple


# 预计算的坐标系统 - 所有可能的航点坐标 (一次性生成，快速查找)
PRECOMPUTED_COORDINATES = {}

def _generate_precomputed_coordinates():
    """一次性生成所有坐标，存储在全局字典中"""
    global PRECOMPUTED_COORDINATES
    if PRECOMPUTED_COORDINATES:  # 如果已经生成过，直接返回
        return PRECOMPUTED_COORDINATES
    
    coordinates = {}
    # 生成所有A1B1到A9B7的航点
    for row in range(1, 8):  # B1 到 B7
        for col in range(1, 10):  # A1 到 A9
            # 从 A1 到 A9 对应 x 坐标从 4.0 到 0.0 (递减)
            x = (9 - col) * 0.5
            # 从 B1 到 B7 对应 y 坐标从 0.0 到 3.0
            y = (row - 1) * 0.5
            point_name = f"A{col}B{row}"
            coordinates[point_name] = (x, y)
    
    # 添加特殊标记点 A9A9 (任务完成标志)
    coordinates["A9A9"] = (0.0, 0.0)
    
    PRECOMPUTED_COORDINATES = coordinates
    return coordinates

def generate_coordinate_system():
    """返回预计算的坐标系统字典"""
    return _generate_precomputed_coordinates()

class OptimizedWaypointManager:
    """优化的航点管理器 - 确保快速解析和单次处理"""
    
    def __init__(self):
        # 预加载所有坐标
        self.coordinates = _generate_precomputed_coordinates()
        
        # 跟踪已处理的航点 (确保每个航点只进行一次图像识别)
        self.processed_waypoints = set()
        
        # 统计信息
        self.total_waypoints = len([wp for wp in self.coordinates.keys() if wp != "A9A9"])
        self.processed_count = 0
        
    def get_coordinate(self, waypoint_name: str) -> tuple:
        """O(1)时间复杂度获取航点坐标"""
        return self.coordinates.get(waypoint_name, None)
    
    def is_valid_waypoint(self, waypoint_name: str) -> bool:
        """检查是否为有效航点"""
        return waypoint_name in self.coordinates
    
    def is_completion_marker(self, waypoint_name: str) -> bool:
        """检查是否为任务完成标记"""
        return waypoint_name == "A9A9"
    
    def should_process_waypoint(self, waypoint_name: str) -> bool:
        """检查航点是否应该处理 (未处理过且非完成标记)"""
        return (waypoint_name not in self.processed_waypoints and 
                not self.is_completion_marker(waypoint_name) and
                self.is_valid_waypoint(waypoint_name))
    
    def mark_waypoint_processed(self, waypoint_name: str):
        """标记航点已处理，避免重复图像识别"""
        if waypoint_name not in self.processed_waypoints:
            self.processed_waypoints.add(waypoint_name)
            self.processed_count += 1
            print(f"✅ 航点 {waypoint_name} 已标记为已处理 ({self.processed_count}/{self.total_waypoints})")
    
    def get_processing_stats(self):
        """获取处理统计信息"""
        return {
            'total_waypoints': self.total_waypoints,
            'processed_count': self.processed_count,
            'remaining_count': self.total_waypoints - self.processed_count,
            'completion_percentage': (self.processed_count / self.total_waypoints) * 100 if self.total_waypoints > 0 else 0
        }
    
    def batch_validate_waypoints(self, waypoint_list: list) -> dict:
        """批量验证航点列表，返回验证结果"""
        result = {
            'valid_waypoints': [],
            'invalid_waypoints': [],
            'already_processed': [],
            'completion_markers': []
        }
        
        for waypoint in waypoint_list:
            if not self.is_valid_waypoint(waypoint):
                result['invalid_waypoints'].append(waypoint)
            elif self.is_completion_marker(waypoint):
                result['completion_markers'].append(waypoint)
            elif waypoint in self.processed_waypoints:
                result['already_processed'].append(waypoint)
            else:
                result['valid_waypoints'].append(waypoint)
        
        return result
    
    def get_all_waypoint_names(self) -> list:
        """获取所有航点名称列表（排除完成标记）"""
        return [wp for wp in self.coordinates.keys() if wp != "A9A9"]
    
    def reset_processing_state(self):
        """重置处理状态 (用于任务重启)"""
        self.processed_waypoints.clear()
        self.processed_count = 0
        print("🔄 航点处理状态已重置")

# 创建全局优化航点管理器实例
OPTIMIZED_WAYPOINT_MANAGER = OptimizedWaypointManager()