#!/usr/bin/env python3
"""
摄像头画面截图工具
支持多种截图模式、图像处理、批量管理等功能
"""

import cv2
import numpy as np
import argparse
import os
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import json

@dataclass
class ScreenshotConfig:
    """截图配置"""
    output_dir: str = "screenshots"
    filename_prefix: str = "screenshot"
    image_format: str = "jpg"
    quality: int = 95
    add_timestamp: bool = True
    add_frame_number: bool = False
    resize_factor: float = 1.0
    apply_filter: str = "none"

class CameraScreenshotTool:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.cap = None
        self.config = ScreenshotConfig()
        
        # 截图统计
        self.screenshot_count = 0
        self.session_start_time = datetime.now()
        
        # 运动检测参数
        self.motion_threshold = 30
        self.motion_min_area = 500
        self.background_subtractor = None
        self.last_motion_time = 0
        self.motion_cooldown = 2.0  # 秒
        
        # 定时截图参数
        self.interval_mode = False
        self.interval_seconds = 5.0
        self.last_interval_time = 0
        
        # 连续截图参数
        self.burst_mode = False
        self.burst_count = 5
        self.burst_interval = 0.5
        
        # 图像处理选项
        self.available_filters = {
            'none': self._filter_none,
            'blur': self._filter_blur,
            'sharpen': self._filter_sharpen,
            'emboss': self._filter_emboss,
            'edge': self._filter_edge,
            'sepia': self._filter_sepia,
            'vintage': self._filter_vintage
        }
        
        # 窗口名称
        self.window_name = f"Camera Screenshot Tool - Device {device_id}"
        
        # 支持的图像格式
        self.supported_formats = {
            'jpg': '.jpg',
            'jpeg': '.jpeg',
            'png': '.png',
            'bmp': '.bmp',
            'tiff': '.tiff',
            'webp': '.webp'
        }

    def open_camera(self) -> bool:
        """打开摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                print(f"无法打开摄像头 {self.device_id}")
                return False
            
            # 设置合理的分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # 初始化运动检测
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
            
            print(f"摄像头 {self.device_id} 已打开")
            return True
            
        except Exception as e:
            print(f"打开摄像头时发生错误：{e}")
            return False

    def close_camera(self):
        """关闭摄像头"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def ensure_output_directory(self):
        """确保输出目录存在"""
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
            print(f"创建输出目录: {self.config.output_dir}")

    def generate_filename(self, frame_number: Optional[int] = None) -> str:
        """生成文件名"""
        components = [self.config.filename_prefix]
        
        if self.config.add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 毫秒精度
            components.append(timestamp)
        
        if self.config.add_frame_number and frame_number is not None:
            components.append(f"frame_{frame_number:06d}")
        
        self.screenshot_count += 1
        components.append(f"{self.screenshot_count:04d}")
        
        filename = "_".join(components) + self.supported_formats[self.config.image_format]
        return os.path.join(self.config.output_dir, filename)

    def apply_image_processing(self, frame: np.ndarray) -> np.ndarray:
        """应用图像处理"""
        processed = frame.copy()
        
        # 调整大小
        if self.config.resize_factor != 1.0:
            new_width = int(frame.shape[1] * self.config.resize_factor)
            new_height = int(frame.shape[0] * self.config.resize_factor)
            processed = cv2.resize(processed, (new_width, new_height))
        
        # 应用滤镜
        if self.config.apply_filter in self.available_filters:
            processed = self.available_filters[self.config.apply_filter](processed)
        
        return processed

    def add_overlay_info(self, frame: np.ndarray) -> np.ndarray:
        """添加覆盖信息"""
        overlay_frame = frame.copy()
        
        # 半透明背景
        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (10, frame.shape[0] - 80), (400, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, overlay_frame, 0.3, 0, overlay_frame)
        
        # 添加文本信息
        info_lines = [
            f"Screenshots: {self.screenshot_count}",
            f"Format: {self.config.image_format.upper()} Q:{self.config.quality}",
            f"Filter: {self.config.apply_filter.title()}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = frame.shape[0] - 60 + i * 20
            cv2.putText(overlay_frame, line, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay_frame

    def save_screenshot(self, frame: np.ndarray, frame_number: Optional[int] = None) -> str:
        """保存截图"""
        self.ensure_output_directory()
        
        # 处理图像
        processed_frame = self.apply_image_processing(frame)
        
        # 生成文件名
        filename = self.generate_filename(frame_number)
        
        # 设置保存参数
        save_params = []
        if self.config.image_format in ['jpg', 'jpeg']:
            save_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.quality]
        elif self.config.image_format == 'png':
            # PNG压缩级别 (0-9)
            compression_level = int((100 - self.config.quality) / 100 * 9)
            save_params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
        elif self.config.image_format == 'webp':
            save_params = [cv2.IMWRITE_WEBP_QUALITY, self.config.quality]
        
        # 保存图像
        try:
            success = cv2.imwrite(filename, processed_frame, save_params)
            if success:
                file_size = os.path.getsize(filename) / 1024  # KB
                print(f"截图已保存: {filename} ({file_size:.1f} KB)")
                return filename
            else:
                print(f"保存截图失败: {filename}")
                return ""
        except Exception as e:
            print(f"保存截图时发生错误: {e}")
            return ""

    def detect_motion(self, frame: np.ndarray) -> bool:
        """检测运动"""
        if self.background_subtractor is None:
            return False
        
        # 应用背景减法
        fg_mask = self.background_subtractor.apply(frame)
        
        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 检查是否有足够大的运动区域
        for contour in contours:
            if cv2.contourArea(contour) > self.motion_min_area:
                return True
        
        return False

    def take_burst_screenshots(self, frame: np.ndarray):
        """连续截图"""
        print(f"开始连续截图，共 {self.burst_count} 张...")
        
        for i in range(self.burst_count):
            # 获取新的帧
            ret, current_frame = self.cap.read()
            if ret:
                filename = self.save_screenshot(current_frame, i + 1)
                if filename:
                    print(f"连续截图 {i + 1}/{self.burst_count}: {os.path.basename(filename)}")
            
            if i < self.burst_count - 1:  # 不是最后一张
                time.sleep(self.burst_interval)
        
        print("连续截图完成")

    def run_interactive_mode(self):
        """运行交互模式"""
        if not self.cap:
            print("摄像头未打开")
            return
        
        print("截图工具已启动")
        print("按键控制:")
        print("  SPACE: 单张截图")
        print("  b: 连续截图模式")
        print("  i: 切换定时截图")
        print("  m: 切换运动检测截图")
        print("  1-7: 切换图像滤镜")
        print("  +/-: 调整图像质量")
        print("  r: 调整缩放比例")
        print("  f: 切换图像格式")
        print("  c: 显示配置信息")
        print("  s: 显示统计信息")
        print("  q: 退出")
        
        frame_counter = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break
                
                frame_counter += 1
                current_time = time.time()
                
                # 运动检测截图
                if hasattr(self, 'motion_detection_mode') and self.motion_detection_mode:
                    if self.detect_motion(frame):
                        if current_time - self.last_motion_time > self.motion_cooldown:
                            print("检测到运动，自动截图")
                            self.save_screenshot(frame, frame_counter)
                            self.last_motion_time = current_time
                
                # 定时截图
                if self.interval_mode:
                    if current_time - self.last_interval_time >= self.interval_seconds:
                        print("定时自动截图")
                        self.save_screenshot(frame, frame_counter)
                        self.last_interval_time = current_time
                
                # 添加信息覆盖
                display_frame = self.add_overlay_info(frame)
                
                # 显示状态指示
                status_indicators = []
                if hasattr(self, 'motion_detection_mode') and self.motion_detection_mode:
                    status_indicators.append("MOTION")
                if self.interval_mode:
                    status_indicators.append(f"INTERVAL({self.interval_seconds}s)")
                
                if status_indicators:
                    status_text = " | ".join(status_indicators)
                    cv2.putText(display_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(self.window_name, display_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # 单张截图
                    self.save_screenshot(frame, frame_counter)
                elif key == ord('b'):
                    # 连续截图
                    self.take_burst_screenshots(frame)
                elif key == ord('i'):
                    # 切换定时截图
                    self.interval_mode = not self.interval_mode
                    if self.interval_mode:
                        self.last_interval_time = current_time
                        print(f"定时截图已开启，间隔 {self.interval_seconds} 秒")
                    else:
                        print("定时截图已关闭")
                elif key == ord('m'):
                    # 切换运动检测
                    if not hasattr(self, 'motion_detection_mode'):
                        self.motion_detection_mode = False
                    self.motion_detection_mode = not self.motion_detection_mode
                    
                    if self.motion_detection_mode:
                        print("运动检测截图已开启")
                    else:
                        print("运动检测截图已关闭")
                elif key >= ord('1') and key <= ord('7'):
                    # 切换滤镜
                    filters = list(self.available_filters.keys())
                    filter_index = key - ord('1')
                    if filter_index < len(filters):
                        self.config.apply_filter = filters[filter_index]
                        print(f"切换到滤镜: {self.config.apply_filter}")
                elif key == ord('+') or key == ord('='):
                    # 提高质量
                    self.config.quality = min(100, self.config.quality + 5)
                    print(f"图像质量: {self.config.quality}")
                elif key == ord('-'):
                    # 降低质量
                    self.config.quality = max(10, self.config.quality - 5)
                    print(f"图像质量: {self.config.quality}")
                elif key == ord('r'):
                    # 切换缩放比例
                    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
                    current_index = scales.index(self.config.resize_factor) if self.config.resize_factor in scales else 2
                    next_index = (current_index + 1) % len(scales)
                    self.config.resize_factor = scales[next_index]
                    print(f"缩放比例: {self.config.resize_factor:.2f}")
                elif key == ord('f'):
                    # 切换图像格式
                    formats = list(self.supported_formats.keys())
                    current_index = formats.index(self.config.image_format) if self.config.image_format in formats else 0
                    next_index = (current_index + 1) % len(formats)
                    self.config.image_format = formats[next_index]
                    print(f"图像格式: {self.config.image_format.upper()}")
                elif key == ord('c'):
                    # 显示配置
                    self.print_config()
                elif key == ord('s'):
                    # 显示统计
                    self.print_statistics()
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")

    def print_config(self):
        """打印当前配置"""
        print("\n=== 当前配置 ===")
        print(f"输出目录: {self.config.output_dir}")
        print(f"文件名前缀: {self.config.filename_prefix}")
        print(f"图像格式: {self.config.image_format.upper()}")
        print(f"图像质量: {self.config.quality}")
        print(f"缩放比例: {self.config.resize_factor:.2f}")
        print(f"应用滤镜: {self.config.apply_filter}")
        print(f"添加时间戳: {'是' if self.config.add_timestamp else '否'}")
        print(f"添加帧编号: {'是' if self.config.add_frame_number else '否'}")

    def print_statistics(self):
        """打印统计信息"""
        current_time = datetime.now()
        session_duration = current_time - self.session_start_time
        
        print("\n=== 截图统计 ===")
        print(f"会话开始时间: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"会话持续时间: {session_duration}")
        print(f"总截图数量: {self.screenshot_count}")
        
        if session_duration.total_seconds() > 0:
            rate = self.screenshot_count / session_duration.total_seconds() * 60
            print(f"平均截图频率: {rate:.2f} 张/分钟")
        
        # 统计输出目录大小
        if os.path.exists(self.config.output_dir):
            total_size = 0
            file_count = 0
            for filename in os.listdir(self.config.output_dir):
                filepath = os.path.join(self.config.output_dir, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
                    file_count += 1
            
            print(f"输出目录文件数: {file_count}")
            print(f"输出目录总大小: {total_size / (1024*1024):.2f} MB")

    # 滤镜函数
    def _filter_none(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def _filter_blur(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, (15, 15), 0)

    def _filter_sharpen(self, frame: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(frame, -1, kernel)

    def _filter_emboss(self, frame: np.ndarray) -> np.ndarray:
        kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
        return cv2.filter2D(frame, -1, kernel)

    def _filter_edge(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def _filter_sepia(self, frame: np.ndarray) -> np.ndarray:
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        return cv2.transform(frame, kernel)

    def _filter_vintage(self, frame: np.ndarray) -> np.ndarray:
        # 添加暖色调和轻微模糊
        sepia = self._filter_sepia(frame)
        blurred = cv2.GaussianBlur(sepia, (3, 3), 0)
        return cv2.addWeighted(sepia, 0.7, blurred, 0.3, 0)

    def batch_process_directory(self, input_dir: str, output_dir: str):
        """批量处理目录中的图像"""
        if not os.path.exists(input_dir):
            print(f"输入目录不存在: {input_dir}")
            return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        processed_count = 0
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                input_path = os.path.join(input_dir, filename)
                
                try:
                    # 读取图像
                    frame = cv2.imread(input_path)
                    if frame is not None:
                        # 应用处理
                        processed_frame = self.apply_image_processing(frame)
                        
                        # 保存处理后的图像
                        output_path = os.path.join(output_dir, f"processed_{filename}")
                        cv2.imwrite(output_path, processed_frame)
                        
                        processed_count += 1
                        print(f"处理完成: {filename}")
                
                except Exception as e:
                    print(f"处理 {filename} 时出错: {e}")
        
        print(f"批量处理完成，共处理 {processed_count} 个文件")


def main():
    parser = argparse.ArgumentParser(description="摄像头画面截图工具")
    parser.add_argument('--device', type=int, default=0, help='摄像头设备编号，默认0')
    parser.add_argument('--output-dir', type=str, default='screenshots', help='输出目录，默认screenshots')
    parser.add_argument('--prefix', type=str, default='screenshot', help='文件名前缀，默认screenshot')
    parser.add_argument('--format', type=str, default='jpg', 
                       choices=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
                       help='图像格式，默认jpg')
    parser.add_argument('--quality', type=int, default=95, help='图像质量(1-100)，默认95')
    parser.add_argument('--scale', type=float, default=1.0, help='缩放比例，默认1.0')
    parser.add_argument('--filter', type=str, default='none',
                       choices=['none', 'blur', 'sharpen', 'emboss', 'edge', 'sepia', 'vintage'],
                       help='图像滤镜，默认none')
    parser.add_argument('--interval', type=float, help='定时截图间隔（秒）')
    parser.add_argument('--motion', action='store_true', help='启用运动检测截图')
    parser.add_argument('--burst', type=int, help='连续截图数量')
    parser.add_argument('--burst-interval', type=float, default=0.5, help='连续截图间隔（秒）')
    parser.add_argument('--batch-process', nargs=2, metavar=('INPUT_DIR', 'OUTPUT_DIR'),
                       help='批量处理模式：输入目录 输出目录')
    
    args = parser.parse_args()

    tool = CameraScreenshotTool(args.device)
    
    # 配置设置
    tool.config.output_dir = args.output_dir
    tool.config.filename_prefix = args.prefix
    tool.config.image_format = args.format
    tool.config.quality = args.quality
    tool.config.resize_factor = args.scale
    tool.config.apply_filter = args.filter
    
    if args.interval:
        tool.interval_mode = True
        tool.interval_seconds = args.interval
    
    if args.motion:
        tool.motion_detection_mode = True
    
    if args.burst:
        tool.burst_count = args.burst
        tool.burst_interval = args.burst_interval
    
    try:
        if args.batch_process:
            # 批量处理模式
            input_dir, output_dir = args.batch_process
            tool.batch_process_directory(input_dir, output_dir)
        else:
            # 正常截图模式
            if not tool.open_camera():
                sys.exit(1)
            
            if args.burst:
                # 连续截图模式
                print("准备连续截图...")
                ret, frame = tool.cap.read()
                if ret:
                    tool.take_burst_screenshots(frame)
            else:
                # 交互模式
                tool.run_interactive_mode()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误：{e}")
    finally:
        tool.close_camera()


if __name__ == "__main__":
    main()