#!/usr/bin/env python3
"""
摄像头实时参数调节工具
提供GUI界面实时调节摄像头各种参数
"""

import cv2
import argparse
import json
import os
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class CameraParameters:
    """摄像头参数配置"""
    brightness: float = 0.5
    contrast: float = 0.5
    saturation: float = 0.5
    hue: float = 0.5
    gain: float = 0.5
    exposure: float = 0.5
    white_balance: float = 0.5
    focus: float = 0.5
    gamma: float = 0.5
    sharpness: float = 0.5

class CameraRealtimeAdjuster:
    def __init__(self, device_id: int = 0, config_file: str = "camera_config.json"):
        self.device_id = device_id
        self.config_file = config_file
        self.cap = None
        self.window_name = f"Camera {device_id} - Real-time Adjuster"
        self.control_window = f"Camera {device_id} Controls"
        
        # 参数映射表 (OpenCV属性ID, 显示名称, 范围, 默认值)
        self.parameter_map = {
            'brightness': (cv2.CAP_PROP_BRIGHTNESS, "Brightness", (-1.0, 1.0), 0.0),
            'contrast': (cv2.CAP_PROP_CONTRAST, "Contrast", (-1.0, 1.0), 0.0),
            'saturation': (cv2.CAP_PROP_SATURATION, "Saturation", (-1.0, 1.0), 0.0),
            'hue': (cv2.CAP_PROP_HUE, "Hue", (-1.0, 1.0), 0.0),
            'gain': (cv2.CAP_PROP_GAIN, "Gain", (0.0, 1.0), 0.0),
            'exposure': (cv2.CAP_PROP_EXPOSURE, "Exposure", (-10.0, 0.0), -5.0),
            'white_balance': (cv2.CAP_PROP_WB_TEMPERATURE, "White Balance", (2000.0, 8000.0), 4000.0),
            'focus': (cv2.CAP_PROP_FOCUS, "Focus", (0.0, 1.0), 0.5),
            'gamma': (cv2.CAP_PROP_GAMMA, "Gamma", (0.5, 3.0), 1.0),
            'sharpness': (cv2.CAP_PROP_SHARPNESS, "Sharpness", (0.0, 1.0), 0.5),
        }
        
        # 当前参数值（trackbar值，0-100范围）
        self.current_values = {}
        
        # 保存原始参数值用于重置
        self.original_values = {}

    def open_camera(self) -> bool:
        """打开摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                print(f"无法打开摄像头 {self.device_id}")
                return False
            
            print(f"摄像头 {self.device_id} 已打开")
            
            # 保存原始参数值
            self._save_original_values()
            
            return True
            
        except Exception as e:
            print(f"打开摄像头时发生错误：{e}")
            return False

    def close_camera(self):
        """关闭摄像头"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def _save_original_values(self):
        """保存原始参数值"""
        for param_name, (prop_id, _, _, _) in self.parameter_map.items():
            try:
                value = self.cap.get(prop_id)
                self.original_values[param_name] = value
            except:
                self.original_values[param_name] = 0

    def _trackbar_to_actual(self, param_name: str, trackbar_value: int) -> float:
        """将trackbar值(0-100)转换为实际参数值"""
        _, _, (min_val, max_val), _ = self.parameter_map[param_name]
        return min_val + (max_val - min_val) * (trackbar_value / 100.0)

    def _actual_to_trackbar(self, param_name: str, actual_value: float) -> int:
        """将实际参数值转换为trackbar值(0-100)"""
        _, _, (min_val, max_val), _ = self.parameter_map[param_name]
        normalized = (actual_value - min_val) / (max_val - min_val)
        return int(max(0, min(100, normalized * 100)))

    def _create_trackbars(self):
        """创建控制滑块"""
        cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.control_window, 400, 600)
        
        # 为每个参数创建滑块
        for param_name, (prop_id, display_name, _, default_val) in self.parameter_map.items():
            # 获取当前值或使用默认值
            try:
                current_actual = self.cap.get(prop_id)
                if current_actual == -1:  # 参数不支持
                    current_actual = default_val
            except:
                current_actual = default_val
            
            # 转换为trackbar值
            trackbar_value = self._actual_to_trackbar(param_name, current_actual)
            self.current_values[param_name] = trackbar_value
            
            # 创建滑块
            cv2.createTrackbar(
                display_name, 
                self.control_window, 
                trackbar_value, 
                100, 
                lambda val, name=param_name: self._on_trackbar_change(name, val)
            )
        
        # 添加特殊按钮（通过trackbar实现）
        cv2.createTrackbar("Reset All", self.control_window, 0, 1, self._on_reset_all)
        cv2.createTrackbar("Save Config", self.control_window, 0, 1, self._on_save_config)
        cv2.createTrackbar("Load Config", self.control_window, 0, 1, self._on_load_config)

    def _on_trackbar_change(self, param_name: str, trackbar_value: int):
        """trackbar变化回调"""
        if not self.cap:
            return
        
        self.current_values[param_name] = trackbar_value
        actual_value = self._trackbar_to_actual(param_name, trackbar_value)
        
        prop_id, display_name, _, _ = self.parameter_map[param_name]
        
        try:
            # 设置参数
            self.cap.set(prop_id, actual_value)
            
            # 验证设置结果
            set_value = self.cap.get(prop_id)
            print(f"{display_name}: {actual_value:.3f} (实际: {set_value:.3f})")
            
        except Exception as e:
            print(f"设置 {display_name} 失败: {e}")

    def _on_reset_all(self, value: int):
        """重置所有参数"""
        if value == 0:
            return
        
        print("重置所有参数到原始值...")
        
        for param_name, original_value in self.original_values.items():
            prop_id, display_name, _, _ = self.parameter_map[param_name]
            
            try:
                # 设置为原始值
                self.cap.set(prop_id, original_value)
                
                # 更新trackbar
                trackbar_value = self._actual_to_trackbar(param_name, original_value)
                cv2.setTrackbarPos(display_name, self.control_window, trackbar_value)
                
            except Exception as e:
                print(f"重置 {display_name} 失败: {e}")
        
        # 重置按钮自身
        cv2.setTrackbarPos("Reset All", self.control_window, 0)

    def _on_save_config(self, value: int):
        """保存配置"""
        if value == 0:
            return
        
        try:
            config = {}
            for param_name, trackbar_value in self.current_values.items():
                actual_value = self._trackbar_to_actual(param_name, trackbar_value)
                config[param_name] = actual_value
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"配置已保存到 {self.config_file}")
            
        except Exception as e:
            print(f"保存配置失败: {e}")
        
        # 重置按钮
        cv2.setTrackbarPos("Save Config", self.control_window, 0)

    def _on_load_config(self, value: int):
        """加载配置"""
        if value == 0:
            return
        
        try:
            if not os.path.exists(self.config_file):
                print(f"配置文件 {self.config_file} 不存在")
                return
            
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            print(f"从 {self.config_file} 加载配置...")
            
            for param_name, actual_value in config.items():
                if param_name in self.parameter_map:
                    prop_id, display_name, _, _ = self.parameter_map[param_name]
                    
                    try:
                        # 设置参数
                        self.cap.set(prop_id, actual_value)
                        
                        # 更新trackbar
                        trackbar_value = self._actual_to_trackbar(param_name, actual_value)
                        cv2.setTrackbarPos(display_name, self.control_window, trackbar_value)
                        self.current_values[param_name] = trackbar_value
                        
                    except Exception as e:
                        print(f"加载 {display_name} 失败: {e}")
            
        except Exception as e:
            print(f"加载配置失败: {e}")
        
        # 重置按钮
        cv2.setTrackbarPos("Load Config", self.control_window, 0)

    def _draw_parameter_info(self, frame):
        """在画面上绘制参数信息"""
        y_pos = 30
        line_height = 25
        
        # 背景半透明矩形
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, y_pos + len(self.current_values) * line_height + 10), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制参数文本
        for param_name, trackbar_value in self.current_values.items():
            _, display_name, _, _ = self.parameter_map[param_name]
            actual_value = self._trackbar_to_actual(param_name, trackbar_value)
            
            text = f"{display_name}: {actual_value:.3f}"
            cv2.putText(frame, text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_pos += line_height

    def run(self, show_info: bool = True):
        """运行实时调节界面"""
        if not self.cap:
            print("摄像头未打开")
            return
        
        # 创建控制面板
        self._create_trackbars()
        
        print(f"实时参数调节器已启动")
        print("使用右侧控制面板调节参数")
        print("按 'q' 退出, 'r' 重置, 's' 保存配置, 'l' 加载配置")
        print("按 'h' 切换参数信息显示")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 在画面上显示参数信息
            if show_info:
                self._draw_parameter_info(frame)
            
            # 显示画面
            cv2.imshow(self.window_name, frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._on_reset_all(1)
            elif key == ord('s'):
                self._on_save_config(1)
            elif key == ord('l'):
                self._on_load_config(1)
            elif key == ord('h'):
                show_info = not show_info
            elif key == ord('i'):
                self._print_current_values()

    def _print_current_values(self):
        """打印当前所有参数值"""
        print("\n=== 当前参数值 ===")
        for param_name, trackbar_value in self.current_values.items():
            _, display_name, _, _ = self.parameter_map[param_name]
            actual_value = self._trackbar_to_actual(param_name, trackbar_value)
            
            # 获取实际设置的值
            prop_id = self.parameter_map[param_name][0]
            try:
                current_actual = self.cap.get(prop_id)
                print(f"{display_name:15}: {actual_value:8.3f} (实际: {current_actual:8.3f})")
            except:
                print(f"{display_name:15}: {actual_value:8.3f} (实际: 获取失败)")

    def get_supported_parameters(self) -> Dict[str, bool]:
        """获取摄像头支持的参数"""
        supported = {}
        
        for param_name, (prop_id, display_name, _, _) in self.parameter_map.items():
            try:
                # 尝试获取参数值
                value = self.cap.get(prop_id)
                supported[param_name] = value != -1
            except:
                supported[param_name] = False
        
        return supported

    def print_camera_info(self):
        """打印摄像头信息"""
        if not self.cap:
            print("摄像头未打开")
            return
        
        print(f"\n=== 摄像头 {self.device_id} 信息 ===")
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        backend = self.cap.getBackendName()
        
        print(f"分辨率: {width}x{height}")
        print(f"帧率: {fps:.1f} fps")
        print(f"后端: {backend}")
        
        print("\n=== 支持的参数 ===")
        supported = self.get_supported_parameters()
        for param_name, is_supported in supported.items():
            _, display_name, _, _ = self.parameter_map[param_name]
            status = "✓" if is_supported else "✗"
            print(f"{status} {display_name}")


def main():
    parser = argparse.ArgumentParser(description="摄像头实时参数调节工具")
    parser.add_argument('--device', type=int, default=0, help='摄像头设备编号，默认0')
    parser.add_argument('--config', type=str, default='camera_config.json', help='配置文件路径')
    parser.add_argument('--info', action='store_true', help='显示摄像头信息')
    parser.add_argument('--no-overlay', action='store_true', help='不在画面上显示参数信息')
    
    args = parser.parse_args()

    adjuster = CameraRealtimeAdjuster(args.device, args.config)
    
    try:
        if not adjuster.open_camera():
            sys.exit(1)
        
        if args.info:
            adjuster.print_camera_info()
        
        # 运行实时调节器
        adjuster.run(show_info=not args.no_overlay)
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误：{e}")
    finally:
        adjuster.close_camera()


if __name__ == "__main__":
    main()