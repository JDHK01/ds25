import cv2
import numpy as np
from typing import List, Dict, Callable, Tuple
from functools import wraps

# --------------------- Detection Visualizer ---------------------

class DetectionVisualizer:
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
                 display_delay: int = 1):
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

    def visualize(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        vis_frame = frame.copy()
        for i, det in enumerate(detections):
            center = det.get('center', None)
            bbox = det.get('bbox', None)
            area = det.get('area', None)

            if self.show_bbox and bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2),
                              self.bbox_color, self.bbox_thickness)

            if self.show_center and center is not None:
                cv2.circle(vis_frame, center, self.center_radius,
                           self.center_color, -1)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
                label = f"#{i+1}"
                if self.show_area and area is not None:
                    label += f" Area: {int(area)}"
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale, self.text_thickness)
                cv2.rectangle(vis_frame,
                              (x1, text_y - text_height - 4),
                              (x1 + text_width + 4, text_y + 4),
                              self.bbox_color, -1)
                cv2.putText(vis_frame, label, (x1 + 2, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
                            self.text_color, self.text_thickness)

        info_text = f"Detected: {len(detections)} objects"
        cv2.putText(vis_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        return vis_frame

    def display(self, frame: np.ndarray) -> None:
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(self.display_delay)

    def wrap_detector(self, detect_func: Callable) -> Callable:
        @wraps(detect_func)
        def wrapper(obj_self, frame: np.ndarray) -> List[Dict]:
            detections = detect_func(obj_self, frame)
            vis_frame = self.visualize(frame, detections)
            if self.auto_display:
                self.display(vis_frame)
            if hasattr(obj_self, '_last_vis_frame'):
                obj_self._last_vis_frame = vis_frame
            return detections
        return wrapper


def visualize_detections(window_name: str = "Detection Results", **kwargs) -> Callable:
    visualizer = DetectionVisualizer(window_name=window_name, **kwargs)
    def decorator(func: Callable) -> Callable:
        return visualizer.wrap_detector(func)
    return decorator

# --------------------- QR Code Detector ---------------------

class QRCodeDetector:
    def __init__(self, min_area: int = 1):
        self.min_area = min_area

    @visualize_detections(
        window_name="QR Code Detection",
        bbox_color=(255, 0, 0),
        center_color=(0, 255, 255),
        show_area=True,
        show_center=True,
        show_bbox=True,
        auto_display=True,
        display_delay=1
    )
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        qr_detector = cv2.QRCodeDetector()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []
        retval, points = qr_detector.detectMulti(gray)
        if retval and points is not None:
            for qr_points in points:
                qr_points = qr_points.astype(int)
                x_coords = qr_points[:, 0]
                y_coords = qr_points[:, 1]
                x = np.min(x_coords)
                y = np.min(y_coords)
                w = np.max(x_coords) - x
                h = np.max(y_coords) - y
                center_x = x + w // 2
                center_y = y + h // 2
                area = w * h
                if area > self.min_area:
                    detections.append({
                        'center': (center_x, center_y),
                        'bbox': (x, y, x + w, y + h),
                        'area': area
                    })
        return detections

# --------------------- 摄像头实时检测 ---------------------

def run_camera_detection(camera_index: int = 6):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头编号 {camera_index}")
        return

    print(f"📷 正在打开摄像头 {camera_index}，按 'q' 键退出")
    detector = QRCodeDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取帧")
            break

        detector.detect_objects(frame)

        # 按下 q 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --------------------- 启动 ---------------------

if __name__ == "__main__":
    run_camera_detection(camera_index=6)
