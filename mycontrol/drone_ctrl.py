import sys
sys.path.append("/home/by/ds25/temp/lib")
sys.path.append("/home/by/ds25/temp/vision/yolo")
sys.path.append("/home/by/ds25/temp/util")
from ser import * 
from detect import *
from camera_device_scanner import CameraScanner

import asyncio
import math
import cv2
import numpy as np
import time
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed
import Jetson.GPIO as GPIO
import LED_Flash as led

from mavsdk.offboard import ( PositionNedYaw, VelocityBodyYawspeed)


# --- 串口设置 ---
DRONERECEIVE = '#'
DRONESEND = '$ANI'
LORA_PACKET_FOOTER = "%"
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600

# ===============================跟踪相关的枚举和类型定义================================
class TargetMode(Enum):
    """目标定位模式"""
    DOWN = "down"      # 摄像头向下，垂直对准目标

class TaskState(Enum):
    """任务执行状态"""
    TRACKING = "tracking"      # 跟踪目标中
    APPROACHING = "approaching" # 靠近目标中
    COMPLETED = "completed"    # 跟踪完成

@dataclass
class CameraConfig:
    """相机参数配置"""
    width: int = 640
    height: int = 480
    fps: int = 30
    device_id: int = 0
    buffer_size: int = 1
    
    # 相机相对于无人机中心的偏移(m):前右下
    offset_forward: float = 0.0   # 相机在无人机前方的距离（正值表示在前）
    offset_right: float = 0.0     # 相机在无人机右侧的距离（正值表示在右）
    offset_down: float = 0.0      # 相机在无人机下方的距离（正值表示在下）
    
    # 显示配置
    show_window: bool = False      # 是否显示窗口界面（调试时开启，正式运行时关闭）

@dataclass
class DroneCommand:
    """无人机机体坐标系速度命令"""
    velocity_forward: float = 0.0  # 向前速度 (m/s)
    velocity_right: float = 0.0    # 向右速度 (m/s)
    velocity_down: float = 0.0     # 向下速度 (m/s)

class PIDController:
    """PID控制器"""
    def __init__(self, kp: float = 0.3, ki: float = 0.0, kd: float = 0.0,
                 output_limit: float = 1.0, integral_limit: float = 5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None
        
    def compute(self, error: float) -> float:
        """计算PID输出"""
        current_time = time.time()
        
        if self.last_time is None:
            dt = 0.02
        else:
            dt = current_time - self.last_time
            if dt <= 0:
                dt = 0.02
        
        # P项
        p_term = self.kp * error
        
        # I项（带抗积分饱和）
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # D项
        d_term = self.kd * (error - self.prev_error) / dt if dt > 0 else 0
        
        # 总输出
        output = p_term + i_term + d_term
        output = np.clip(output, -self.output_limit, self.output_limit)
        
        # 更新历史
        self.prev_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """重置控制器"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None

class Drone_Controller:
    def __init__(self, path_label=["A9B1"], camera_config=None, navigation_config=None, pid_config=None):
        """
        初始化函数,将路径标签转换为NED坐标路径点。
        添加视觉跟踪功能。
        """
        self.path_label = path_label
        
        # 初始化相机配置
        self.camera_config = camera_config or CameraConfig()
        
        # 导航参数
        nav_config = navigation_config or {}
        self.position_tolerance = nav_config.get('position_tolerance', 30)
        self.min_target_area = nav_config.get('min_target_area', 1000)
        self.max_velocity = nav_config.get('max_velocity', 0.5)
        self.offset_compensation_gain = nav_config.get('offset_compensation_gain', 1.0)
        self.alignment_duration = nav_config.get('alignment_duration', 2.0)
        self.completion_tolerance = nav_config.get('completion_tolerance', 15)
        
        # PID控制器配置
        pid_config = pid_config or {}
        self.pid_x = PIDController(**pid_config.get('horizontal', {}))
        self.pid_y = PIDController(**pid_config.get('horizontal', {}))
        
        # 跟踪状态
        self.task_state = TaskState.TRACKING
        self.is_tracking = False
        self.last_detection_time = 0
        self.detection_timeout = 2.0
        self.alignment_start_time = None
        self.task_completed = False
        self.completion_time = None
        
        # 相机对象（将在需要时初始化）
        self.camera = None
        
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

        self.visit_status = {label: 0 for label in self.label_map.keys()}

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

    async def goto_next(self, drone, current_point, next_point, ser_port, cap, detector, label=None):
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
        #第一次遍历到时进行动物识别，并串口发送识别结果
        if self.visit_status[label] == 0:
            self.visit_status[label] = 1
            # 清空缓冲区，获取最新帧
            for _ in range(2):
                cap.grab()
            ret, frame = cap.retrieve()
            if ret:
                # 视野裁切：从左上角(180,50)到(540,410)
                cropped_frame = frame[50:410, 180:540]
                result = detector.detect_animals(cropped_frame, show_result=False)
                if not result:
                    print("未识别到")
                else:
                    print("识别到")
                    ser_port.send_lora_packet(DRONESEND ,label + self.format_animal_counts(result), footer=LORA_PACKET_FOOTER)
            else:
                print("摄像头读取失败")    
            
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

        flash_task = None
        stop_event = asyncio.Event()

        print("[降落] A8B1逻辑")
        (x, y) = self.label_map["A8B1"]
        await self.goto_position_ned(drone, x, y, -0.5, 0.0, 5)#duiration 待测
        await self.print_current_position(drone)
        flash_task = asyncio.create_task(led.led_flash(stop_event))
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(-0.5, 0, 0.5, 0))  # 45度速度控制
        async for pos in drone.telemetry.position_velocity_ned():
            if -pos.position.down_m < 0.12:
                print("[降落] 已触地")
                await drone.action.kill()
                break
        print("[降落] 结束闪烁")
        stop_event.set()
        if flash_task:
            await flash_task  # 等待闪烁任务退出

    async def land_from_A9B2(self, drone):
        
        flash_task = None
        stop_event = asyncio.Event()

        print("[降落] A9B2逻辑")
        (x, y) = self.label_map["A9B2"]
        await self.goto_position_ned(drone, x, y, -0.5, 0.0, 5)
        await self.print_current_position(drone)
        flash_task = asyncio.create_task(led.led_flash(stop_event))
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, -0.5, 0.5, 0))  # 45度速度控制
        async for pos in drone.telemetry.position_velocity_ned():
            if -pos.position.down_m < 0.12:
                print("[降落] 已触地")
                await drone.action.kill()
                break
        print("[降落] 结束闪烁")  
        stop_event.set()
        if flash_task:
            await flash_task  # 等待闪烁任务退出

    def format_animal_counts(self, animal_dict):#转换检测结果为数据包格式
        # 严格定义顺序：elephant(e) → monkey(m) → peacock(p) → wolf(w) → tiger(t)
            order = [
                ('elephant', 'e'),
                ('monkey', 'm'),
                ('peacock', 'p'),
                ('wolf', 'w'),
                ('tiger', 't')
            ]
        # 按每个动物的数量，不存在则为0
            parts = []
            for animal, abbr in order:
                count = animal_dict.get(animal, 0)
                parts.append(f"{abbr}{count}")
        
        # 拼接成最终字符串
            return ''.join(parts)
    
    # ===============================视觉跟踪相关方法================================
    def _find_available_camera(self):
        """自动查找可用的摄像头设备"""
        scanner = CameraScanner(max_devices=5)
        devices = scanner.scan_devices(verbose=False)
        
        if not devices:
            print("警告: 未找到任何可用摄像头")
            return 0  # 返回默认设备ID
        
        # 优先选择设备ID较小的摄像头
        available_device = min(devices, key=lambda d: d.device_id)
        print(f"自动选择摄像头设备ID: {available_device.device_id}")
        return available_device.device_id

    def init_camera(self):
        """初始化相机"""
        if self.camera is None:
            # 如果配置的设备ID不可用，自动查找可用设备
            device_id = self.camera_config.device_id
            cap_test = cv2.VideoCapture(device_id)
            if not cap_test.isOpened():
                cap_test.release()
                print(f"摄像头设备ID {device_id} 不可用，正在自动查找...")
                device_id = self._find_available_camera()
                self.camera_config.device_id = device_id
            else:
                cap_test.release()
            
            self.camera = cv2.VideoCapture(device_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.camera_config.fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.camera_config.buffer_size)
            print(f"相机初始化成功 - 设备ID: {device_id}")
    
    def compute_pixel_error(self, target_center: Tuple[int, int]) -> Tuple[float, float]:
        """计算目标中心与画面中心的像素误差"""
        frame_center_x = self.camera_config.width // 2
        frame_center_y = self.camera_config.height // 2
        
        # 归一化误差 (-1 到 1)
        error_x = (target_center[0] - frame_center_x) / (self.camera_config.width // 2)
        error_y = (target_center[1] - frame_center_y) / (self.camera_config.height // 2)
        
        return error_x, error_y
    
    def compute_tracking_command(self, detection: Dict) -> DroneCommand:
        """根据检测结果计算跟踪控制命令"""
        center_x, center_y = detection['center']
        error_x, error_y = self.compute_pixel_error((center_x, center_y))
        
        command = DroneCommand()
        
        # 基础控制量
        base_right = self.pid_x.compute(error_x)
        base_forward = -self.pid_y.compute(error_y)  # y轴反向
        
        # 偏移补偿
        if abs(error_x) < 0.1 and abs(error_y) < 0.1:
            # 接近中心时，加入偏移补偿
            command.velocity_forward = base_forward - self.camera_config.offset_forward * self.offset_compensation_gain
            command.velocity_right = base_right - self.camera_config.offset_right * self.offset_compensation_gain
        else:
            # 远离中心时，正常控制
            command.velocity_forward = base_forward
            command.velocity_right = base_right
        
        command.velocity_down = 0.0
            
        # 速度限制
        command.velocity_forward = np.clip(command.velocity_forward, -self.max_velocity, self.max_velocity)
        command.velocity_right = np.clip(command.velocity_right, -self.max_velocity, self.max_velocity)
        command.velocity_down = np.clip(command.velocity_down, -self.max_velocity, self.max_velocity)
        
        return command
    
    def process_tracking_frame(self, detector) -> Tuple[Optional[np.ndarray], Optional[DroneCommand]]:
        """处理相机帧并生成跟踪控制命令"""
        if self.camera is None:
            self.init_camera()
        
        # 读取帧
        ret, frame = self.camera.read()
        if not ret:
            return None, None
        
        # 如果任务已完成，只返回画面，不生成命令
        if self.task_state == TaskState.COMPLETED:
            return frame, DroneCommand(0.0, 0.0, 0.0)

        # 视野裁切：从左上角(180,50)到(540,410)
        cropped_frame = frame[50:410, 180:540]
        
        # 使用传入的检测器进行检测
        result = detector.detect_animals(cropped_frame, show_result=False)
        command = None
        
        # 将检测结果转换为适合跟踪的格式
        detections = []
        if result:
            # 假设检测器返回的是动物检测结果，转换为跟踪格式
            # 这里需要根据实际的detector.detect_animals返回格式进行调整
            for animal_type, boxes in result.items():
                for box in boxes:
                    # 假设box格式为 [x1, y1, x2, y2, confidence]
                    if len(box) >= 4:
                        x1, y1, x2, y2 = box[:4]
                        # 将裁切区域的坐标转换回原始帧坐标
                        x1 += 180
                        y1 += 50
                        x2 += 180
                        y2 += 50
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        area = (x2 - x1) * (y2 - y1)
                        
                        if area > self.min_target_area:
                            detections.append({
                                'center': (center_x, center_y),
                                'bbox': (x1, y1, x2, y2),
                                'area': area,
                                'class_name': animal_type,
                                'confidence': box[4] if len(box) > 4 else 0.0
                            })
        
        # 检测到物体 -> 处理跟踪
        if detections:
            # 选择距离画面中心最近的目标
            if len(detections) == 1:
                best_detection = detections[0]
            else:
                frame_center_x = self.camera_config.width // 2
                frame_center_y = self.camera_config.height // 2
                
                min_distance = float('inf')
                best_detection = None
                
                for detection in detections:
                    center_x, center_y = detection['center']
                    distance = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        best_detection = detection
                        
            # 处理选中的目标
            center_x, center_y = best_detection['center']
            error_x, error_y = self.compute_pixel_error((center_x, center_y))
            pixel_error = np.sqrt(error_x**2 + error_y**2) * (self.camera_config.width // 2)
            
            # 更新跟踪状态
            self.is_tracking = True
            self.last_detection_time = time.time()
            
            # 状态转换逻辑
            if pixel_error <= self.completion_tolerance:
                if self.task_state == TaskState.TRACKING:
                    self.task_state = TaskState.APPROACHING
                    self.alignment_start_time = time.time()
                    print(f"状态转换: TRACKING -> APPROACHING")
                
                # 检查是否完成靠近
                if self.task_state == TaskState.APPROACHING:
                    if time.time() - self.alignment_start_time >= self.alignment_duration:
                        self.task_state = TaskState.COMPLETED
                        self.task_completed = True
                        self.completion_time = time.time()
                        command = DroneCommand(0.0, 0.0, 0.0)  # 停止移动
                        print(f"状态转换: APPROACHING -> COMPLETED")
                        print(f"跟踪完成！保持对准 {self.alignment_duration} 秒")
                    else:
                        # 继续保持位置
                        command = DroneCommand(0.0, 0.0, 0.0)
            else:
                # 需要继续调整位置
                if self.task_state == TaskState.APPROACHING:
                    self.task_state = TaskState.TRACKING
                    self.alignment_start_time = None
                    print(f"状态转换: APPROACHING -> TRACKING (位置偏离)")
                
                command = self.compute_tracking_command(best_detection)
                
        else:
            # 无检测 - 返回到跟踪状态寻找目标
            if self.task_state != TaskState.COMPLETED:
                if time.time() - self.last_detection_time > self.detection_timeout:
                    self.task_state = TaskState.TRACKING
                    self.is_tracking = False
                    self.alignment_start_time = None
                    print(f"目标丢失，返回跟踪状态")
                    # 重置PID控制器
                    self.pid_x.reset()
                    self.pid_y.reset()
        
        return frame, command
    
    def get_task_state(self) -> TaskState:
        """获取当前任务状态"""
        return self.task_state
    
    def is_task_completed(self) -> bool:
        """检查任务是否完成"""
        return self.task_state == TaskState.COMPLETED
    
    def reset_tracking_task(self):
        """重置跟踪任务状态"""
        self.task_state = TaskState.TRACKING
        self.task_completed = False
        self.completion_time = None
        self.alignment_start_time = None
        self.is_tracking = False
        self.pid_x.reset()
        self.pid_y.reset()
        print("跟踪任务状态已重置")
    
    def get_tracking_info(self) -> Dict:
        """获取跟踪任务详细信息"""
        info = {
            'state': self.task_state.value,
            'completed': self.task_completed,
            'tracking': self.is_tracking,
            'completion_time': self.completion_time
        }
        if self.task_state == TaskState.APPROACHING and self.alignment_start_time:
            elapsed = time.time() - self.alignment_start_time
            info['approaching_progress'] = min(elapsed / self.alignment_duration, 1.0)
        return info
    
    def cleanup_camera(self):
        """清理相机资源"""
        if self.camera and self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()
    
    async def visual_tracking_mode(self, drone, detector, duration=10.0):
        """视觉跟踪模式 - 独立的跟踪功能"""
        print(f"[跟踪] 启动视觉跟踪模式，持续时间: {duration}秒")
        
        # 初始化相机
        self.init_camera()
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # 处理跟踪帧
                frame, command = self.process_tracking_frame(detector)
                
                if frame is not None:
                    # 检查任务是否完成
                    if self.is_task_completed():
                        print("视觉跟踪任务已完成！")
                        # 悬停
                        await drone.offboard.set_velocity_body(
                            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                        )
                        return True  # 返回True表示成功完成跟踪
                        
                    elif command is not None:
                        # 发送跟踪速度命令到无人机
                        await drone.offboard.set_velocity_body(
                            VelocityBodyYawspeed(
                                command.velocity_forward,
                                command.velocity_right,
                                command.velocity_down,
                                0.0  # 不调整偏航
                            )
                        )
                        # 格式化输出跟踪信息
                        elapsed_time = time.time() - start_time
                        print(f"[跟踪 {elapsed_time:6.2f}s] 前进:{command.velocity_forward:+6.3f} 右移:{command.velocity_right:+6.3f} 下降:{command.velocity_down:+6.3f}")
                    else:
                        # 悬停
                        await drone.offboard.set_velocity_body(
                            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                        )
                        elapsed_time = time.time() - start_time
                        print(f"[跟踪 {elapsed_time:6.2f}s] 悬停 - 寻找目标")
                
                await asyncio.sleep(0.02)  # 50Hz控制频率
                
        except KeyboardInterrupt:
            print("\n[跟踪] 停止跟踪...")
        finally:
            # 停止移动
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            )
            
        print(f"[跟踪] 跟踪模式结束，持续时间: {time.time() - start_time:.2f}秒")
        return False  # 返回False表示未完成跟踪（超时）

    async def pilot_plan(self, drone, ser_port):

        #=============检测器初始化=============
        # 创建检测器
        detector = YOLOv8AnimalDetector('/home/by/ds25/temp/vision/yolo/best9999.onnx')
        
        # 自动查找可用摄像头设备
        device_id = 0  # 默认设备ID
        cap_test = cv2.VideoCapture(device_id)
        if not cap_test.isOpened():
            cap_test.release()
            print(f"摄像头设备ID {device_id} 不可用，正在自动查找...")
            device_id = self._find_available_camera()
        else:
            cap_test.release()
        
        # 打开摄像头
        cap = cv2.VideoCapture(device_id)
        print(f"使用摄像头设备ID: {device_id}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 设置缓冲区为1帧
        cap.set(cv2.CAP_PROP_FPS, 15)  # 降低帧率减少延迟
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 使用MJPEG编码

        print("[任务] 启动飞行任务")
        if not self.path:
            print("路径为空，无法执行")
            return

        current_point = self.path[0]
        for i in range(1, len(self.path)):
            next_point = self.path[i]
            if current_point == self.label_map["A8B1"]  and next_point == self.label_map["A9B1"]:
                await self.land_from_A8B1(drone)
                return
            elif current_point == self.label_map["A9B2"] and next_point == self.label_map["A9B1"]:
                await self.land_from_A9B2(drone)
                return
            await self.goto_next(drone, current_point, next_point,ser_port, cap, detector, label=self.path_label[i])
            current_point = next_point