#!/usr/bin/env python3
"""
室内无人机视觉导航系统
- 纯视觉反馈
- 基于机体坐标系控制
- 摄像头向下垂直对准目标
"""

import cv2
import numpy as np
import time
import asyncio
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw, VelocityBodyYawspeed)
from mavsdk.telemetry import LandedState
# ===============================常量:枚举类型的================================
class TargetMode(Enum):
    """目标定位模式"""
    DOWN = "down"      # 摄像头向下，垂直对准目标


class TaskState(Enum):
    """任务执行状态"""
    TRACKING = "tracking"      # 跟踪目标中
    APPROACHING = "approaching" # 靠近目标中
    COMPLETED = "completed"    # 跟踪完成

# =============================变量: 存储数据的类=================================
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


# ============================真正的类===================================
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



class ObjectDetector:
    """简化的目标检测器"""
    def __init__(self, detection_config: dict = None):
        '''
        传入的参数:
            探测的配置: 字典
            并从中特别取出min_area
        '''
        self.detection_config = detection_config or {}
        self.min_area = self.detection_config.get('min_area', 500)
        
    # def detect_objects(self, frame: np.ndarray) -> List[Dict]:
    #     """使用颜色检测目标"""
    #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     # 红色检测
    #     lower_red1 = np.array([0, 100, 100])
    #     upper_red1 = np.array([10, 255, 255])
    #     lower_red2 = np.array([170, 100, 100])
    #     upper_red2 = np.array([180, 255, 255])
    #     mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    #     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    #     mask = cv2.bitwise_or(mask1, mask2)
    #     # 形态学处理
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #     # 查找轮廓
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     detections = []
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         if area > self.min_area:
    #             x, y, w, h = cv2.boundingRect(contour)
    #             center_x = x + w // 2
    #             center_y = y + h // 2
                
    #             detections.append({
    #                 'center': (center_x, center_y),
    #                 'bbox': (x, y, x + w, y + h),
    #                 'area': area
    #             })
    #     # 返回的是列表, 列表中的每个元素都是字典, 存储每个检测对象的信息:检测的区域, 面积
    #     return detections
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """检测多个二维码位置"""
        # 创建二维码检测器
        qr_detector = cv2.QRCodeDetector()
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detections = []
        
        # 检测多个二维码
        retval, points = qr_detector.detectMulti(gray)
        
        if retval and points is not None:
            for qr_points in points:
                # 将浮点数转换为整数
                qr_points = qr_points.astype(int)
                
                # 计算边界框
                x_coords = qr_points[:, 0]
                y_coords = qr_points[:, 1]
                x = np.min(x_coords)
                y = np.min(y_coords)
                w = np.max(x_coords) - x
                h = np.max(y_coords) - y
                
                # 计算中心点
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 计算面积
                area = w * h
                
                if area > self.min_area:
                    detections.append({
                        'center': (center_x, center_y),
                        'bbox': (x, y, x + w, y + h),
                        'area': area
                    })
        
        return detections

# 视觉导航系统
class VisionGuidanceSystem:
    """视觉导航系统"""
    def __init__(self, camera_config: CameraConfig, target_mode: TargetMode = TargetMode.DOWN,
                 navigation_config: dict = None, pid_config: dict = None):
        '''
        传入的参数:
            对准模式: 向下, 向前
            相机配置参数
            导航参数
            pid控制器参数
        '''
        # 打开相机
        self.camera_config = camera_config
        self.target_mode = target_mode
        self.camera = cv2.VideoCapture(camera_config.device_id)
        self._configure_camera()

        # 初始化检测器
        self.detector = ObjectDetector()

        # 导航参数: 逐个尝试读取, 没有就赋值
        nav_config = navigation_config or {}
        self.position_tolerance = nav_config.get('position_tolerance', 30)        # 像素容差: 目前是90度的视野角, 50cm范围, 640像素宽度, 1pixel -> 0.07cm
        self.min_target_area = nav_config.get('min_target_area', 1000)     # 最小目标面积
        self.max_velocity = nav_config.get('max_velocity', 0.5)            # 最大速度
        self.offset_compensation_gain = nav_config.get('offset_compensation_gain', 1.0)  # 偏移补偿增益
        self.alignment_duration = nav_config.get('alignment_duration', 2.0)  # 对准保持时间
        self.completion_tolerance = nav_config.get('completion_tolerance', 15)  # 完成任务的像素容差
        
        # PID配置参数: 逐个尝试读取, 没有就赋值
        pid_config = pid_config or {}
        # 摄像头向下模式：控制前后左右
        self.pid_x = PIDController(**pid_config.get('horizontal', {}))
        self.pid_y = PIDController(**pid_config.get('horizontal', {}))
        self.pid_z = None  # 不需要高度控制
        
        # 跟踪状态
        self.task_state = TaskState.TRACKING
        self.is_tracking = False
        self.last_detection_time = 0
        self.detection_timeout = 2.0
        self.alignment_start_time = None
        self.task_completed = False
        self.completion_time = None
        print(f"视觉导航系统初始化成功 - 模式: {target_mode.value}")
        
    def _configure_camera(self):
        """配置相机参数"""
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.height)
        self.camera.set(cv2.CAP_PROP_FPS, self.camera_config.fps)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.camera_config.buffer_size)
    
    def compute_pixel_error(self, target_center: Tuple[int, int]) -> Tuple[float, float]:
        """计算目标中心与画面中心的像素误差"""
        frame_center_x = self.camera_config.width // 2
        frame_center_y = self.camera_config.height // 2
        
        # 归一化误差 (-1 到 1)
        error_x = (target_center[0] - frame_center_x) / (self.camera_config.width // 2)
        error_y = (target_center[1] - frame_center_y) / (self.camera_config.height // 2)
        
        return error_x, error_y
    
    def compute_control_command(self, detection: Dict) -> DroneCommand:
        """根据检测结果计算控制命令（考虑相机偏移）"""
        center_x, center_y = detection['center']
        error_x, error_y = self.compute_pixel_error((center_x, center_y))
        
        command = DroneCommand()
        
        # 摄像头向下：误差映射到前后左右运动
        # 需要补偿水平偏移
        
        # 基础控制量
        '''
        非常重要的转换:
            假设物体在画面左侧, error_x < 0, 需要左移, 已经对齐
            假设物体在画面上方, error_y < 0, 需要前移动, 未对齐
        '''
        base_right = self.pid_x.compute(error_x)
        base_forward = -self.pid_y.compute(error_y)  # y轴反向
        
        # 偏移补偿
        # 当目标在画面中心时，无人机需要移动到让其中心对准目标
        # 考虑相机偏移：如果相机在无人机前方，需要额外后退
        
        # 判断是否接近目标中心
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
    
    def process_frame(self) -> Tuple[Optional[np.ndarray], Optional[DroneCommand]]:
        """处理相机帧并生成控制命令"""
        # 读取帧
        ret, frame = self.camera.read()
        if not ret:
            return None, None
        
        # 如果任务已完成，只返回画面，不生成命令
        if self.task_state == TaskState.COMPLETED:
            self._draw_status(frame)
            return frame, DroneCommand(0.0, 0.0, 0.0)

        # 对读取到的帧进行处理, 检测目标
        detections = self.detector.detect_objects(frame)
        command = None
        # 检测到物体 -> 选择最大的物体进行跟踪
        if detections:
            # 数据预处理, 为跟踪做准备
            best_detection = max(detections, key=lambda d: d['area'])
            center_x, center_y = best_detection['center']
            error_x, error_y = self.compute_pixel_error((center_x, center_y))
            pixel_error = np.sqrt(error_x**2 + error_y**2) * (self.camera_config.width // 2)
            
            # 更新跟踪状态
            self.is_tracking = True
            self.last_detection_time = time.time()
            
            # 简化的状态转换逻辑
            # 检查是否进入靠近状态
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
                
                command = self.compute_control_command(best_detection)
            
            # 绘制检测结果
            self._draw_detection(frame, best_detection, command)
            
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
        
        # 绘制状态信息
        self._draw_status(frame)
        
        return frame, command
    
    def _draw_detection(self, frame: np.ndarray, detection: Dict, command: Optional[DroneCommand]):
        """在画面上绘制检测结果"""
        x1, y1, x2, y2 = detection['bbox']
        center_x, center_y = detection['center']
        
        # 绘制边界框
        color = (0, 255, 0) if self.is_tracking else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制中心点
        cv2.circle(frame, (center_x, center_y), 5, color, -1)
        
        # 绘制画面中心十字
        frame_center_x = self.camera_config.width // 2
        frame_center_y = self.camera_config.height // 2
        cv2.line(frame, (frame_center_x - 20, frame_center_y), 
                 (frame_center_x + 20, frame_center_y), (0, 255, 255), 2)
        cv2.line(frame, (frame_center_x, frame_center_y - 20), 
                 (frame_center_x, frame_center_y + 20), (0, 255, 255), 2)
        
        # 绘制连接线
        cv2.line(frame, (center_x, center_y), (frame_center_x, frame_center_y), (255, 255, 0), 1)
        
        # 显示信息
        if command:
            info_text = f"Cmd: F:{command.velocity_forward:.2f} R:{command.velocity_right:.2f} D:{command.velocity_down:.2f}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示误差
        error_x, error_y = self.compute_pixel_error((center_x, center_y))
        error_text = f"Error: X:{error_x:.2f} Y:{error_y:.2f}"
        cv2.putText(frame, error_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示偏移补偿状态
        if abs(error_x) < 0.1 and abs(error_y) < 0.1:
            cv2.putText(frame, "Offset Compensation: ON", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示靠近进度
        if self.task_state == TaskState.APPROACHING and self.alignment_start_time:
            elapsed = time.time() - self.alignment_start_time
            progress = min(elapsed / self.alignment_duration, 1.0)
            progress_text = f"Approaching: {progress*100:.0f}%"
            cv2.putText(frame, progress_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def _draw_status(self, frame: np.ndarray):
        """绘制系统状态"""
        # 任务状态
        state_color = {
            TaskState.TRACKING: (0, 255, 255),
            TaskState.APPROACHING: (255, 255, 0),
            TaskState.COMPLETED: (0, 255, 0)
        }
        
        status_text = f"State: {self.task_state.value.upper()}"
        color = state_color.get(self.task_state, (255, 255, 255))
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 模式
        mode_text = f"Mode: {self.target_mode.value.upper()}"
        cv2.putText(frame, mode_text, (frame.shape[1] - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 相机偏移信息
        offset_text = f"Cam Offset: F:{self.camera_config.offset_forward:.2f} R:{self.camera_config.offset_right:.2f} D:{self.camera_config.offset_down:.2f}"
        cv2.putText(frame, offset_text, (10, frame.shape[0] - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 任务完成标记
        if self.task_state == TaskState.COMPLETED:
            cv2.putText(frame, "TASK COMPLETED!", (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    def get_task_state(self) -> TaskState:
        """获取当前任务状态"""
        return self.task_state
    
    def is_task_completed(self) -> bool:
        """检查任务是否完成"""
        return self.task_state == TaskState.COMPLETED
    
    def reset_task(self):
        """重置任务状态，准备新任务"""
        self.task_state = TaskState.TRACKING
        self.task_completed = False
        self.completion_time = None
        self.alignment_start_time = None
        self.is_tracking = False
        self.pid_x.reset()
        self.pid_y.reset()
        print("任务状态已重置")
    
    def get_task_info(self) -> Dict:
        """获取任务详细信息"""
        info = {
            'state': self.task_state.value,
            'completed': self.task_completed,
            'tracking': self.is_tracking,
            'completion_time': self.completion_time,
            'mode': self.target_mode.value
        }
        if self.task_state == TaskState.APPROACHING and self.alignment_start_time:
            elapsed = time.time() - self.alignment_start_time
            info['approaching_progress'] = min(elapsed / self.alignment_duration, 1.0)
        return info
    
    def cleanup(self):
        """清理资源"""
        if self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()


# 异步无人机控制示例
async def drone_control_loop(vision_system: VisionGuidanceSystem, drone):
    """异步无人机控制循环（带2秒超时）"""
    start_time = time.time()
    timeout_duration = 2.0
    
    try:        
        while True:
            # 检查超时
            if time.time() - start_time >= timeout_duration:
                print(f"跟踪超时 ({timeout_duration}秒)，退出函数")
                # 发送停止命令
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                )
                return
            frame, command = vision_system.process_frame()
            if frame is not None:
                if vision_system.camera_config.show_window:
                    cv2.imshow("Vision Guidance", frame)
                
                # 检查任务是否完成
                if vision_system.is_task_completed():
                    print("视觉导航任务已完成！")
                    # 悬停一下
                    await drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                    )
                    
                    print("任务完成")
                    return
                    # 或者重置任务，寻找新目标
                    # vision_system.reset_task()
                    # continue
                    
                elif command is not None:
                    # 发送速度命令到无人机
                    await drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(
                            command.velocity_forward,
                            command.velocity_right,
                            command.velocity_down,
                            0.0  # 不调整偏航
                        )
                    )
                    # 格式化输出时间和速度信息
                    elapsed_time = time.time() - start_time
                    print(f"[{elapsed_time:6.2f}s] 前进:{command.velocity_forward:+6.3f} 右移:{command.velocity_right:+6.3f} 下降:{command.velocity_down:+6.3f}")
                else:
                    # 悬停
                    await drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                    )
                    # 悬停时也输出信息
                    elapsed_time = time.time() - start_time
                    print(f"[{elapsed_time:6.2f}s] 前进:{0.0:+6.3f} 右移:{0.0:+6.3f} 下降:{0.0:+6.3f} [悬停]")   
            # 检查退出
            if vision_system.camera_config.show_window:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # 不显示窗口时用简单的睡眠代替
                pass
                
            await asyncio.sleep(0.02)  # 50Hz控制频率
            
    except KeyboardInterrupt:
        print("\n停止控制...")
    finally:
        vision_system.cleanup()


# 高级任务执行示例
async def execute_multi_phase_mission(vision_system: VisionGuidanceSystem, drone):
    """执行多阶段任务示例（带2秒超时）"""
    phases = [
        {"name": "寻找目标", "mode": TargetMode.DOWN, "next_action": "hover"},
        {"name": "接近目标", "mode": TargetMode.DOWN, "next_action": "land"}
    ]
    
    for phase in phases:
        print(f"\n开始阶段: {phase['name']}")
        
        # 设置模式并重置任务
        vision_system.target_mode = phase['mode']
        vision_system.reset_task()
        
        # 每个阶段2秒超时
        phase_start_time = time.time()
        timeout_duration = 2.0
        
        # 执行视觉导航
        while not vision_system.is_task_completed():
            # 检查阶段超时
            if time.time() - phase_start_time >= timeout_duration:
                print(f"阶段 '{phase['name']}' 超时 ({timeout_duration}秒)，退出")
                # 发送停止命令
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                )
                return
            frame, command = vision_system.process_frame()
            
            if frame is not None:
                if vision_system.camera_config.show_window:
                    cv2.imshow("Vision Guidance", frame)
                
                if command is not None:
                    await drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(
                            command.velocity_forward,
                            command.velocity_right,
                            command.velocity_down,
                            0.0
                        )
                    )
                    # 格式化输出时间和速度信息
                    elapsed_time = time.time() - phase_start_time
                    print(f"[{elapsed_time:6.2f}s] 前进:{command.velocity_forward:+6.3f} 右移:{command.velocity_right:+6.3f} 下降:{command.velocity_down:+6.3f}")
                else:
                    # 悬停时也输出信息
                    elapsed_time = time.time() - phase_start_time
                    print(f"[{elapsed_time:6.2f}s] 前进:{0.0:+6.3f} 右移:{0.0:+6.3f} 下降:{0.0:+6.3f} [悬停]")
                
                # 获取任务状态信息
                task_info = vision_system.get_task_info()
                if task_info['state'] == 'approaching':
                    print(f"靠近进度: {task_info.get('approaching_progress', 0)*100:.0f}%")
            
            if vision_system.camera_config.show_window:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
            else:
                # 不显示窗口时用简单的睡眠代替
                pass
                
            await asyncio.sleep(0.02)
        
        # 阶段完成，执行下一动作
        print(f"阶段 '{phase['name']}' 完成！")
        
        if phase['next_action'] == 'hover':
            print("悬停3秒...")
            await asyncio.sleep(3)
        elif phase['next_action'] == 'land':
            print("降落...")
            await drone.land()
            break


# 主程序示例
if __name__ == "__main__":
    # 相机配置
    camera_config = CameraConfig(
        width=640,
        height=480,
        fps=30,
        device_id=0,  # 根据实际相机调整
        # 相机偏移配置（单位：米）
        offset_forward=0.1,   # 相机在无人机前方10cm
        offset_right=0.0,     # 相机在中心线上
        offset_down=0.05,     # 相机在无人机下方5cm
        # 窗口显示配置
        show_window=False      # 调试时设为True，正式运行时设为False
    )
    
    # 导航配置
    navigation_config = {
        'position_tolerance': 20,    # 像素容差
        'min_target_area': 1000,     # 最小目标面积
        'max_velocity': 0.5,         # 最大速度 m/s
        'offset_compensation_gain': 0.3,  # 偏移补偿增益（0-1）
        'alignment_duration': 2.0,   # 对准保持时间（秒）
        'completion_tolerance': 15   # 完成任务的像素容差
    }
    
    # PID配置
    pid_config = {
        'horizontal': {
            'kp': 0.5,
            'ki': 0.0,
            'kd': 0.1,
            'output_limit': 0.5
        },
        'vertical': {
            'kp': 0.5,
            'ki': 0.0,
            'kd': 0.1,
            'output_limit': 0.5
        },
        'forward': {
            'kp': 0.3,
            'ki': 0.0,
            'kd': 0.05,
            'output_limit': 0.3
        }
    }
    
    # 创建视觉导航系统
    vision_system = VisionGuidanceSystem(
        camera_config=camera_config,
        target_mode=TargetMode.DOWN,
        navigation_config=navigation_config,
        pid_config=pid_config
    )
    
    # 测试运行（无无人机连接，带2秒超时）
    try:
        print("开始视觉导航测试...")
        print("按 'q' 退出, 按 'r' 重置任务")
        print("注意：函数将在2秒后自动退出")
        
        # 添加2秒超时
        test_start_time = time.time()
        timeout_duration = 2.0
        
        while True:
            # 检查超时
            if time.time() - test_start_time >= timeout_duration:
                print(f"\n测试超时 ({timeout_duration}秒)，自动退出")
                break
            frame, command = vision_system.process_frame()
            
            if frame is not None:
                if vision_system.camera_config.show_window:
                    cv2.imshow("Vision Guidance", frame)
                
                if command is not None:
                    # 格式化输出时间和速度信息
                    elapsed_time = time.time() - test_start_time
                    print(f"[{elapsed_time:6.2f}s] 前进:{command.velocity_forward:+6.3f} 右移:{command.velocity_right:+6.3f} 下降:{command.velocity_down:+6.3f}")
                else:
                    # 无命令时也输出信息
                    elapsed_time = time.time() - test_start_time
                    print(f"[{elapsed_time:6.2f}s] 前进:{0.0:+6.3f} 右移:{0.0:+6.3f} 下降:{0.0:+6.3f} [无命令]")
                
                # 显示任务状态
                task_info = vision_system.get_task_info()
                if task_info['state'] != 'tracking':
                    print(f"任务状态: {task_info['state']}", end='')
                    if 'approaching_progress' in task_info:
                        print(f" (进度: {task_info['approaching_progress']*100:.0f}%)", end='')
                    print()
                
                # 检查任务完成
                if vision_system.is_task_completed():
                    print("\n任务完成! 可以执行下一阶段任务")
                    print("按 'r' 重置任务继续寻找目标")
            
            if vision_system.camera_config.show_window:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    vision_system.reset_task()
                    print("任务已重置")
            else:
                # 不显示窗口时使用简单的退出机制（可以通过其他方式控制）
                time.sleep(0.02)
                
    except KeyboardInterrupt:
        print("\n停止...")
    finally:
        vision_system.cleanup()

"""
使用说明：

1. 相机偏移配置：
   - offset_forward: 相机在无人机前方的距离（米），正值表示在前
   - offset_right: 相机在无人机右侧的距离（米），正值表示在右
   - offset_down: 相机在无人机下方的距离（米），正值表示在下

2. 窗口显示配置：
   - show_window: 控制是否显示窗口界面
     * True: 显示窗口（适合调试、测试）
     * False: 不显示窗口（适合正式运行、节省资源）
   - 配置示例：
     camera_config = CameraConfig(show_window=False)  # 关闭窗口显示

3. 偏移补偿工作原理：
   - 当目标接近画面中心时（误差<0.1），系统会自动启用偏移补偿
   - 补偿会让无人机的几何中心（而不是相机中心）对准目标
   - 可通过offset_compensation_gain调整补偿强度（0-1）

4. 调试建议：
   - 调试时设置 show_window=True 查看实时画面
   - 正式运行时设置 show_window=False 提高性能
   - 先将偏移设为0进行测试
   - 逐步加入实际偏移值
   - 观察"Offset Compensation: ON"提示来确认补偿是否生效

5. 任务状态管理（简化版）：
   - TRACKING: 跟踪目标，调整位置
   - APPROACHING: 靠近目标中（保持位置）
   - COMPLETED: 跟踪完成，可执行下一阶段

6. 任务完成条件：
   - 目标在容差范围内（completion_tolerance）
   - 保持对准状态达到指定时间（alignment_duration）
   
7. 集成建议：
   - 使用 is_task_completed() 检查任务是否完成
   - 使用 get_task_info() 获取详细状态信息
   - 使用 reset_task() 重置任务准备新的目标
"""