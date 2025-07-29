class FlightState(Enum):
    """飞行状态"""
    FLYING = "flying"           # 正常飞行
    VISION_NAVIGATION = "vision" # 视觉导航中
    PAUSED = "paused"           # 暂停
    COMPLETED = "completed"     # 完成

@dataclass
class Waypoint:
    """航点数据结构"""
    north: float
    east: float
    down: float
    yaw: float
    duration: float
    name: str = ""

class FlightPathManager:
    """飞行路径管理器"""
    '''
    提供的方法:
        航点索引迭代
        获取航点索引对应的航点信息
        暂停航点飞行
        恢复航点飞行
    '''
    def __init__(self, waypoints: List[Waypoint]):
        '''
        输入:
            航点列表
        '''
        self.waypoints = waypoints# 保存传入的航点列表
        self.current_waypoint_index = 0# 航点的计数器
        self.state = FlightState.FLYING
        # 保留临时退出时的航点信息
        self.paused_position = None
        self.paused_waypoint_index = None
        
    def get_current_waypoint(self) -> Optional[Waypoint]:
        """获取当前目标航点"""
        if self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        return None
    
    def next_waypoint(self):
        """移动到下个航点"""
        if self.current_waypoint_index < len(self.waypoints):
            self.current_waypoint_index += 1
    
    def pause_for_vision_navigation(self, current_position: Tuple[float, float, float, float]):
        """暂停飞行，准备视觉导航"""
        self.state = FlightState.VISION_NAVIGATION# 切换状态
        self.paused_position = current_position
        self.paused_waypoint_index = self.current_waypoint_index
        print(f"暂停飞行，当前位置: {current_position}, 当前航点索引: {self.current_waypoint_index}")
    
    def resume_flight_path(self):
        """恢复飞行路径"""
        self.state = FlightState.FLYING
        print(f"恢复飞行，返回暂停位置: {self.paused_position}")
    
    def is_completed(self) -> bool:
        """检查是否完成所有航点"""
        return self.current_waypoint_index >= len(self.waypoints)
    
    def get_progress(self) -> str:
        """获取进度信息"""
        return f"{self.current_waypoint_index}/{len(self.waypoints)}"