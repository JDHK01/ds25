#!/usr/bin/env python3
"""
摄像头分辨率和参数调节工具
用于测试不同分辨率和调节摄像头参数
"""

import cv2
import argparse
import sys
from typing import List, Tuple, Dict, Optional

class CameraAdjuster:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.cap = None
        self.current_resolution = None
        
        # 常见分辨率列表
        self.common_resolutions = [
            (320, 240),   # QVGA
            (640, 480),   # VGA
            (800, 600),   # SVGA
            (1024, 768),  # XGA
            (1280, 720),  # HD
            (1280, 960),  # 4:3 HD
            (1280, 1024), # SXGA
            (1920, 1080), # Full HD
            (2560, 1440), # QHD
            (3840, 2160), # 4K UHD
        ]
        
        # 摄像头参数映射
        self.camera_properties = {
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'saturation': cv2.CAP_PROP_SATURATION,
            'hue': cv2.CAP_PROP_HUE,
            'gain': cv2.CAP_PROP_GAIN,
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'white_balance': cv2.CAP_PROP_WB_TEMPERATURE,
            'focus': cv2.CAP_PROP_FOCUS,
            'fps': cv2.CAP_PROP_FPS,
        }

    def open_camera(self) -> bool:
        """打开摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                print(f"无法打开摄像头 {self.device_id}")
                return False
            
            # 获取当前分辨率
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.current_resolution = (width, height)
            
            print(f"摄像头 {self.device_id} 已打开，当前分辨率：{width}x{height}")
            return True
            
        except Exception as e:
            print(f"打开摄像头时发生错误：{e}")
            return False

    def close_camera(self):
        """关闭摄像头"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()

    def get_supported_resolutions(self) -> List[Tuple[int, int]]:
        """获取摄像头支持的分辨率"""
        if not self.cap:
            return []
        
        supported = []
        for width, height in self.common_resolutions:
            # 保存当前分辨率
            current_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            current_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 尝试设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # 检查是否设置成功
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_w == width and actual_h == height:
                supported.append((width, height))
            
            # 恢复原分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, current_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, current_h)
        
        return supported

    def set_resolution(self, width: int, height: int) -> bool:
        """设置摄像头分辨率"""
        if not self.cap:
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 验证是否设置成功
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_w == width and actual_h == height:
            self.current_resolution = (width, height)
            print(f"分辨率已设置为：{width}x{height}")
            return True
        else:
            print(f"分辨率设置失败，实际分辨率：{actual_w}x{actual_h}")
            return False

    def get_property_value(self, prop_name: str) -> Optional[float]:
        """获取摄像头属性值"""
        if not self.cap or prop_name not in self.camera_properties:
            return None
        
        prop_id = self.camera_properties[prop_name]
        return self.cap.get(prop_id)

    def set_property_value(self, prop_name: str, value: float) -> bool:
        """设置摄像头属性值"""
        if not self.cap or prop_name not in self.camera_properties:
            return False
        
        prop_id = self.camera_properties[prop_name]
        self.cap.set(prop_id, value)
        
        # 验证设置
        actual_value = self.cap.get(prop_id)
        print(f"{prop_name} 设置为：{value}，实际值：{actual_value}")
        return True

    def get_all_properties(self) -> Dict[str, float]:
        """获取所有摄像头属性"""
        properties = {}
        for prop_name in self.camera_properties:
            value = self.get_property_value(prop_name)
            if value is not None:
                properties[prop_name] = value
        return properties

    def print_camera_info(self):
        """打印摄像头信息"""
        if not self.cap:
            print("摄像头未打开")
            return
        
        print("\n=== 摄像头信息 ===")
        print(f"设备ID: {self.device_id}")
        print(f"当前分辨率: {self.current_resolution[0]}x{self.current_resolution[1]}")
        print(f"帧率: {self.cap.get(cv2.CAP_PROP_FPS):.2f} FPS")
        
        print("\n=== 当前参数 ===")
        properties = self.get_all_properties()
        for prop_name, value in properties.items():
            print(f"{prop_name}: {value:.2f}")

    def show_live_preview(self):
        """显示实时预览"""
        if not self.cap:
            print("摄像头未打开")
            return
        
        print("\n实时预览 - 按以下键进行操作:")
        print("q: 退出预览")
        print("i: 显示摄像头信息")
        print("r: 列出支持的分辨率")
        print("1-9: 快速设置常用分辨率")
        print("b/B: 调节亮度 +/-")
        print("c/C: 调节对比度 +/-")
        print("s/S: 调节饱和度 +/-")
        print("e/E: 调节曝光 +/-")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 在画面上显示当前分辨率和参数信息
            text = f"Resolution: {self.current_resolution[0]}x{self.current_resolution[1]}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示一些关键参数
            brightness = self.get_property_value('brightness')
            contrast = self.get_property_value('contrast')
            if brightness is not None:
                cv2.putText(frame, f"Brightness: {brightness:.2f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if contrast is not None:
                cv2.putText(frame, f"Contrast: {contrast:.2f}", (10, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow(f'Camera {self.device_id} Adjuster', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('i'):
                self.print_camera_info()
            elif key == ord('r'):
                self._list_supported_resolutions()
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                self._set_preset_resolution(key - ord('1'))
            elif key == ord('b'):
                self._adjust_property('brightness', 0.1)
            elif key == ord('B'):
                self._adjust_property('brightness', -0.1)
            elif key == ord('c'):
                self._adjust_property('contrast', 0.1)
            elif key == ord('C'):
                self._adjust_property('contrast', -0.1)
            elif key == ord('s'):
                self._adjust_property('saturation', 0.1)
            elif key == ord('S'):
                self._adjust_property('saturation', -0.1)
            elif key == ord('e'):
                self._adjust_property('exposure', 0.1)
            elif key == ord('E'):
                self._adjust_property('exposure', -0.1)

    def _list_supported_resolutions(self):
        """列出支持的分辨率"""
        print("\n检测支持的分辨率...")
        supported = self.get_supported_resolutions()
        if supported:
            print("支持的分辨率:")
            for i, (w, h) in enumerate(supported):
                print(f"  {i+1}. {w}x{h}")
        else:
            print("未检测到支持的分辨率")

    def _set_preset_resolution(self, index: int):
        """设置预设分辨率"""
        supported = self.get_supported_resolutions()
        if 0 <= index < len(supported):
            width, height = supported[index]
            self.set_resolution(width, height)

    def _adjust_property(self, prop_name: str, delta: float):
        """调节属性值"""
        current = self.get_property_value(prop_name)
        if current is not None:
            new_value = current + delta
            self.set_property_value(prop_name, new_value)


def main():
    parser = argparse.ArgumentParser(description="摄像头分辨率和参数调节工具")
    parser.add_argument('--device', type=int, default=0, help='摄像头设备编号，默认0')
    parser.add_argument('--width', type=int, help='设置分辨率宽度')
    parser.add_argument('--height', type=int, help='设置分辨率高度')
    parser.add_argument('--list-resolutions', action='store_true', help='列出支持的分辨率')
    parser.add_argument('--info', action='store_true', help='显示摄像头信息')
    parser.add_argument('--preview', action='store_true', help='显示实时预览（默认）')
    
    args = parser.parse_args()

    adjuster = CameraAdjuster(args.device)
    
    try:
        if not adjuster.open_camera():
            sys.exit(1)
        
        if args.list_resolutions:
            adjuster._list_supported_resolutions()
        
        if args.info:
            adjuster.print_camera_info()
        
        if args.width and args.height:
            adjuster.set_resolution(args.width, args.height)
        
        # 默认显示预览，除非只是查询信息
        if args.preview or not (args.list_resolutions or args.info):
            adjuster.show_live_preview()
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误：{e}")
    finally:
        adjuster.close_camera()


if __name__ == "__main__":
    main()