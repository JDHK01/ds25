#!/usr/bin/env python3
"""
检测管理器模块
集成YOLO检测系统，提供统一的物体检测接口
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import os
import sys

@dataclass
class DetectionResult:
    """单个检测结果"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]          # (center_x, center_y) 
    confidence: float                # 置信度
    class_id: int                   # 类别ID
    class_name: str                 # 类别名称
    area: int                       # 边界框面积
    timestamp: datetime             # 检测时间戳
    waypoint_position: Optional[Tuple[float, float, float]] = None  # 航点位置(x,y,z)

@dataclass 
class WaypointDetectionInfo:
    """航点检测信息"""
    waypoint_name: str
    waypoint_position: Tuple[float, float, float]  # (x, y, z)
    timestamp: datetime
    detections: List[DetectionResult]
    total_objects: int

class DetectionManager:
    """检测管理器 - 集成YOLO检测和信息存储"""
    
    def __init__(self, model_path: str = "vision/yolo/dump/best.pt", 
                 conf_threshold: float = 0.5, device: str = "cpu",
                 camera_id: int = 0):
        """
        初始化检测管理器
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值
            device: 计算设备
            camera_id: 摄像头ID
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.camera_id = camera_id
        
        # 检测历史记录
        self.detection_history: List[WaypointDetectionInfo] = []
        self.total_detections = 0
        
        # COCO类别名称
        self.class_names =['elephant', 'monkey', 'peacock', 'tiger', 'wolf']
        
        # 初始化YOLO检测器
        self._init_detector()
        
        # 初始化摄像头
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            print(f"警告: 无法打开摄像头 {camera_id}")
            self.camera = None
            
        print(f"检测管理器初始化完成 - 模型: {model_path}, 置信度: {conf_threshold}")
    
    def _init_detector(self):
        """初始化YOLO检测器"""
        try:
            # 尝试导入YOLO
            if os.path.exists(self.model_path):
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.detector_type = "yolo"
                print(f"使用YOLO模型: {self.model_path}")
            else:
                # 如果没有YOLO模型，使用简单的颜色检测作为备选
                self.model = None
                self.detector_type = "simple"
                print("YOLO模型未找到，使用简单检测器")
        except ImportError:
            print("YOLO不可用，使用简单检测器")
            self.model = None
            self.detector_type = "simple"
    
    def detect_objects_from_camera(self, waypoint_name: str = "", 
                                 waypoint_position: Tuple[float, float, float] = (0, 0, 0)) -> List[DetectionResult]:
        """
        从摄像头检测物体
        
        Args:
            waypoint_name: 当前航点名称
            waypoint_position: 当前航点位置
            
        Returns:
            检测结果列表
        """
        if self.camera is None:
            print("摄像头不可用")
            return []
            
        ret, frame = self.camera.read()
        if not ret:
            print("无法读取摄像头帧")
            return []
            
        return self.detect_objects_from_frame(frame, waypoint_name, waypoint_position)
    
    def detect_objects_from_frame(self, frame: np.ndarray, waypoint_name: str = "",
                                waypoint_position: Tuple[float, float, float] = (0, 0, 0)) -> List[DetectionResult]:
        """
        从图像帧检测物体
        
        Args:
            frame: 输入图像帧
            waypoint_name: 当前航点名称  
            waypoint_position: 当前航点位置
            
        Returns:
            检测结果列表
        """
        if self.detector_type == "yolo" and self.model is not None:
            return self._yolo_detect(frame, waypoint_position)
        else:
            return self._simple_detect(frame, waypoint_position)
    
    def _yolo_detect(self, frame: np.ndarray, waypoint_position: Tuple[float, float, float]) -> List[DetectionResult]:
        """使用YOLO进行检测"""
        try:
            results = self.model(frame, conf=self.conf_threshold)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # 计算中心点和面积
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        area = (x2 - x1) * (y2 - y1)
                        
                        # 获取类别名称
                        class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class{cls}"
                        
                        detection = DetectionResult(
                            bbox=(x1, y1, x2, y2),
                            center=(center_x, center_y),
                            confidence=conf,
                            class_id=cls,
                            class_name=class_name,
                            area=area,
                            timestamp=datetime.now(),
                            waypoint_position=waypoint_position
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"YOLO检测出错: {e}")
            return []
    
    def _simple_detect(self, frame: np.ndarray, waypoint_position: Tuple[float, float, float]) -> List[DetectionResult]:
        """简单的颜色检测（备选方案）"""
        try:
            # 转换为HSV色彩空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 检测红色物体作为示例
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # 形态学处理
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # 最小面积阈值
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    detection = DetectionResult(
                        bbox=(x, y, x + w, y + h),
                        center=(center_x, center_y),
                        confidence=0.8,  # 简单检测的固定置信度
                        class_id=0,
                        class_name="red_object",
                        area=area,
                        timestamp=datetime.now(),
                        waypoint_position=waypoint_position
                    )
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"简单检测出错: {e}")
            return []
    
    def save_detection_info(self, waypoint_name: str, waypoint_position: Tuple[float, float, float], 
                          detections: List[DetectionResult]):
        """保存航点检测信息"""
        waypoint_info = WaypointDetectionInfo(
            waypoint_name=waypoint_name,
            waypoint_position=waypoint_position,
            timestamp=datetime.now(),
            detections=detections,
            total_objects=len(detections)
        )
        
        self.detection_history.append(waypoint_info)
        self.total_detections += len(detections)
        
        print(f"保存航点 {waypoint_name} 的检测信息: {len(detections)} 个物体")
    
    def get_largest_detection(self, detections: List[DetectionResult]) -> Optional[DetectionResult]:
        """获取面积最大的检测结果"""
        if not detections:
            return None
        return max(detections, key=lambda d: d.area)
    
    def get_detection_summary(self) -> Dict:
        """获取检测统计摘要"""
        if not self.detection_history:
            return {
                'total_waypoints': 0,
                'total_detections': 0,
                'detection_rate': 0,
                'most_common_class': None
            }
        
        # 统计类别
        class_counts = {}
        for waypoint_info in self.detection_history:
            for detection in waypoint_info.detections:
                class_name = detection.class_name
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        most_common_class = max(class_counts.items(), key=lambda x: x[1]) if class_counts else None
        
        waypoints_with_detections = sum(1 for info in self.detection_history if info.total_objects > 0)
        detection_rate = waypoints_with_detections / len(self.detection_history) if self.detection_history else 0
        
        return {
            'total_waypoints': len(self.detection_history),
            'total_detections': self.total_detections,
            'detection_rate': detection_rate,
            'most_common_class': most_common_class,
            'class_distribution': class_counts
        }
    
    def export_detection_log(self, filename: str = "detection_log.txt"):
        """导出检测日志"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("无人机检测任务日志\n")
                f.write("=" * 50 + "\n")
                f.write(f"任务时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总航点数: {len(self.detection_history)}\n")
                f.write(f"总检测数: {self.total_detections}\n\n")
                
                for i, waypoint_info in enumerate(self.detection_history, 1):
                    f.write(f"航点 {i}: {waypoint_info.waypoint_name}\n")
                    f.write(f"位置: {waypoint_info.waypoint_position}\n")
                    f.write(f"时间: {waypoint_info.timestamp.strftime('%H:%M:%S')}\n")
                    f.write(f"检测数量: {waypoint_info.total_objects}\n")
                    
                    for j, detection in enumerate(waypoint_info.detections, 1):
                        f.write(f"  物体{j}: {detection.class_name} "
                               f"(置信度: {detection.confidence:.2f}, "
                               f"中心: {detection.center}, "
                               f"面积: {detection.area})\n")
                    f.write("\n")
                
                # 添加统计摘要
                summary = self.get_detection_summary()
                f.write("统计摘要\n")
                f.write("-" * 20 + "\n")
                f.write(f"检测率: {summary['detection_rate']:.1%}\n")
                if summary['most_common_class']:
                    f.write(f"最常见类别: {summary['most_common_class'][0]} "
                           f"({summary['most_common_class'][1]}次)\n")
                
            print(f"检测日志已导出到: {filename}")
            
        except Exception as e:
            print(f"导出日志失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        if self.camera and self.camera.isOpened():
            self.camera.release()
        print("检测管理器资源已清理")

# 测试代码
if __name__ == "__main__":
    # 创建检测管理器
    detector = DetectionManager()
    
    # 测试检测功能
    print("开始测试检测功能...")
    
    try:
        # 模拟在几个航点进行检测
        test_waypoints = [
            ("A1B1", (0, 0, -1)),
            ("A2B1", (0.5, 0, -1)),
            ("A3B1", (1.0, 0, -1))
        ]
        
        for waypoint_name, position in test_waypoints:
            print(f"\n在航点 {waypoint_name} 进行检测...")
            detections = detector.detect_objects_from_camera(waypoint_name, position)
            
            if detections:
                print(f"检测到 {len(detections)} 个物体:")
                for i, detection in enumerate(detections, 1):
                    print(f"  {i}. {detection.class_name} - 置信度: {detection.confidence:.2f}")
                
                # 保存检测信息
                detector.save_detection_info(waypoint_name, position, detections)
            else:
                print("未检测到物体")
                detector.save_detection_info(waypoint_name, position, [])
            
            time.sleep(1)  # 模拟飞行间隔
        
        # 显示统计信息
        summary = detector.get_detection_summary()
        print(f"\n任务统计:")
        print(f"总航点: {summary['total_waypoints']}")
        print(f"总检测: {summary['total_detections']}")
        print(f"检测率: {summary['detection_rate']:.1%}")
        
        # 导出日志
        detector.export_detection_log("test_detection_log.txt")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    finally:
        detector.cleanup()