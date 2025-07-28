import cv2
import numpy as np
from typing import List, Dict, Callable, Tuple, Optional
from functools import wraps

class DetectionVisualizer:
    """检测结果可视化器"""
    
    def __init__(self, 
                 window_name: str = "Detection Results",
                 bbox_color: Tuple[int, int, int] = (0, 255, 0),
                 bbox_thickness: int = 2,
                 center_radius: int = 5,
                 center_color: Tuple[int, int, int] = (0, 0, 255),
                 text_color: Tuple[int, int, int] = (255, 255, 255),
                 text_scale: float = 0.6,
                 text_thickness: int = 2,
                 show_area: bool = True,
                 show_center: bool = True,
                 show_bbox: bool = True,
                 auto_display: bool = True,
                 display_delay: int = 1,
                 display_width: Optional[int] = None,
                 display_height: Optional[int] = None):
        """
        初始化可视化器
        
        Args:
            window_name: 显示窗口名称
            bbox_color: 边界框颜色 (B, G, R)
            bbox_thickness: 边界框线宽
            center_radius: 中心点半径
            center_color: 中心点颜色 (B, G, R)
            text_color: 文本颜色 (B, G, R)
            text_scale: 文本大小
            text_thickness: 文本线宽
            show_area: 是否显示面积
            show_center: 是否显示中心点
            show_bbox: 是否显示边界框
            auto_display: 是否自动显示结果
            display_delay: 显示延迟（毫秒），0表示等待按键
            display_width: 显示窗口宽度，None表示使用原始宽度
            display_height: 显示窗口高度，None表示使用原始高度
        """
        self.window_name = window_name
        self.bbox_color = bbox_color
        self.bbox_thickness = bbox_thickness
        self.center_radius = center_radius
        self.center_color = center_color
        self.text_color = text_color
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.show_area = show_area
        self.show_center = show_center
        self.show_bbox = show_bbox
        self.auto_display = auto_display
        self.display_delay = display_delay
        self.display_width = display_width
        self.display_height = display_height
    
    def visualize(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            frame: 输入图像
            detections: 检测结果列表
            
        Returns:
            绘制了检测结果的图像
        """
        # 复制图像，避免修改原图
        vis_frame = frame.copy()
        
        for i, det in enumerate(detections):
            # 获取检测信息
            center = det.get('center', None)
            bbox = det.get('bbox', None)
            area = det.get('area', None)
            
            # 绘制边界框
            if self.show_bbox and bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), 
                            self.bbox_color, self.bbox_thickness)
            
            # 绘制中心点
            if self.show_center and center is not None:
                cv2.circle(vis_frame, center, self.center_radius, 
                          self.center_color, -1)
            
            # 绘制文本信息
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                # 文本位置（在边界框上方）
                text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
                
                # 对象编号
                label = f"#{i+1}"
                
                # 添加面积信息
                if self.show_area and area is not None:
                    label += f" Area: {int(area)}"
                
                # 绘制文本背景
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 
                    self.text_scale, self.text_thickness)
                
                cv2.rectangle(vis_frame, 
                            (x1, text_y - text_height - 4),
                            (x1 + text_width + 4, text_y + 4),
                            self.bbox_color, -1)
                
                # 绘制文本
                cv2.putText(vis_frame, label, (x1 + 2, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
                           self.text_color, self.text_thickness)
        
        # 在左上角显示检测数量
        info_text = f"Detected: {len(detections)} objects"
        cv2.putText(vis_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 255, 255), 2)
        
        return vis_frame
    
    def display(self, frame: np.ndarray) -> None:
        """显示图像"""
        display_frame = frame
        
        # 调整显示尺寸
        if self.display_width is not None or self.display_height is not None:
            h, w = frame.shape[:2]
            
            # 计算新的尺寸
            if self.display_width is not None and self.display_height is not None:
                new_width, new_height = self.display_width, self.display_height
            elif self.display_width is not None:
                # 只指定宽度，按比例计算高度
                new_width = self.display_width
                new_height = int(h * (self.display_width / w))
            else:
                # 只指定高度，按比例计算宽度
                new_height = self.display_height
                new_width = int(w * (self.display_height / h))
            
            # 调整图像尺寸
            display_frame = cv2.resize(frame, (new_width, new_height))
        
        cv2.imshow(self.window_name, display_frame)
        cv2.waitKey(self.display_delay)
    
    def wrap_detector(self, detect_func: Callable) -> Callable:
        """
        包装检测函数，自动添加可视化功能
        
        Args:
            detect_func: 原始检测函数
            
        Returns:
            带可视化功能的检测函数
        """
        @wraps(detect_func)
        def wrapper(obj_self, frame: np.ndarray) -> List[Dict]:
            # 调用原始检测函数
            detections = detect_func(obj_self, frame)
            
            # 可视化结果
            vis_frame = self.visualize(frame, detections)
            
            # 自动显示
            if self.auto_display:
                self.display(vis_frame)
            
            # 保存可视化结果到对象属性（可选）
            if hasattr(obj_self, '_last_vis_frame'):
                obj_self._last_vis_frame = vis_frame
            
            # 返回原始检测结果
            return detections
        
        return wrapper


def visualize_detections(window_name: str = "Detection Results",
                        **kwargs) -> Callable:
    """
    装饰器函数，用于给检测方法添加可视化功能
    
    使用示例:
        @visualize_detections(window_name="QR Code Detection", bbox_color=(255, 0, 0))
        def detect_objects(self, frame: np.ndarray) -> List[Dict]:
            # 你的检测代码
            pass
    """
    visualizer = DetectionVisualizer(window_name=window_name, **kwargs)
    
    def decorator(func: Callable) -> Callable:
        return visualizer.wrap_detector(func)
    
    return decorator


# 独立的可视化函数（不需要装饰器）
def draw_detections(frame: np.ndarray, 
                   detections: List[Dict],
                   **kwargs) -> np.ndarray:
    """
    直接在图像上绘制检测结果
    
    Args:
        frame: 输入图像
        detections: 检测结果
        **kwargs: DetectionVisualizer的参数
        
    Returns:
        绘制了检测结果的图像
    """
    visualizer = DetectionVisualizer(auto_display=False, **kwargs)
    return visualizer.visualize(frame, detections)



class QRDetector:
    def __init__(self):
        self.min_area = 100  # 最小面积阈值
    
    @visualize_detections(
        window_name="QR Code Detection",
        bbox_color=(0, 255, 0),  # 绿色边框
        center_color=(255, 0, 0),  # 蓝色中心点
        show_area=True,
        display_delay=1,  # 1毫秒延迟，连续显示
        display_width=640,  # 可以设置显示宽度
        display_height=480   # 可以设置显示高度
    )
    
    # def detect_objects(self, frame: np.ndarray) -> List[Dict]:
    #     """检测多个二维码位置"""
    #     # 你的原始代码完全不变
    #     qr_detector = cv2.QRCodeDetector()
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     detections = []
        
    #     retval, points = qr_detector.detectMulti(gray)
        
    #     if retval and points is not None:
    #         for qr_points in points:
    #             qr_points = qr_points.astype(int)
    #             x_coords = qr_points[:, 0]
    #             y_coords = qr_points[:, 1]
    #             x = np.min(x_coords)
    #             y = np.min(y_coords)
    #             w = np.max(x_coords) - x
    #             h = np.max(y_coords) - y
    #             center_x = x + w // 2
    #             center_y = y + h // 2
    #             area = w * h
                
    #             if area > self.min_area:
    #                 detections.append({
    #                     'center': (center_x, center_y),
    #                     'bbox': (x, y, x + w, y + h),
    #                     'area': area
    #                 })
        
    #     return detections

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """检测图像中红色区域的位置"""
        # 将图像从 BGR 转换到 HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 定义红色的 HSV 范围（红色在HSV中有两个区段）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # 创建两个掩码并合并
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 查找轮廓
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = x + w // 2
                center_y = y + h // 2

                detections.append({
                    'center': (center_x, center_y),
                    'bbox': (x, y, x + w, y + h),
                    'area': area
                })

        return detections


if __name__ == "__main__":
    # 使用
    detector = QRDetector()
    cap = cv2.VideoCapture(6)  # 或者读取视频文件

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 调用检测函数，自动显示可视化结果
            detections = detector.detect_objects(frame)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"发生异常: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


    
    # 示例1: 使用装饰器
    class Detector:
        def __init__(self):
            self.min_area = 100
        
        @visualize_detections(
            window_name="My Detection",
            bbox_color=(0, 255, 0),
            show_area=True
        )
        def detect_objects(self, frame: np.ndarray) -> List[Dict]:
            # 这里是你的检测代码
            # 返回格式必须是 List[Dict]，每个字典包含:
            # - 'center': (x, y)
            # - 'bbox': (x1, y1, x2, y2)
            # - 'area': float
            return [
                {
                    'center': (100, 100),
                    'bbox': (50, 50, 150, 150),
                    'area': 10000
                }
            ]

    # 示例2: 直接使用可视化器
    def detect_and_visualize(frame, detector):
        # 检测
        detections = detector.detect_objects(frame)
        
        # 可视化
        vis_frame = draw_detections(
            frame, 
            detections,
            bbox_color=(255, 0, 0),
            show_center=True
        )
        
        # 显示
        cv2.imshow("Results", vis_frame)
        cv2.waitKey(0)
        
        return detections

    # 示例3: 使用可视化器类
    def manual_visualize(frame, detector):
        visualizer = DetectionVisualizer(
            bbox_color=(0, 0, 255),
            center_color=(255, 255, 0),
            show_area=True,
            auto_display=True
        )
        
        # 检测
        detections = detector.detect_objects(frame)
        
        # 可视化并显示
        vis_frame = visualizer.visualize(frame, detections)
        visualizer.display(vis_frame)
        
        return detections
    

