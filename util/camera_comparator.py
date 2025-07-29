#!/usr/bin/env python3
"""
摄像头画面对比工具
支持多摄像头同时对比、画质分析、延迟测试等功能
"""

import cv2
import numpy as np
import argparse
import sys
import time
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque

@dataclass
class CameraStats:
    """摄像头统计信息"""
    device_id: int
    fps: float = 0.0
    frame_count: int = 0
    dropped_frames: int = 0
    avg_latency: float = 0.0
    brightness_avg: float = 0.0
    contrast_score: float = 0.0
    last_frame_time: float = 0.0

class CameraComparator:
    def __init__(self, device_ids: List[int]):
        self.device_ids = device_ids
        self.caps = {}
        self.stats = {}
        self.frames = {}
        self.frame_history = {}  # 存储历史帧用于回放
        self.history_size = 100
        
        # 显示参数
        self.display_mode = 'grid'  # 'grid', 'horizontal', 'vertical'
        self.display_size = (640, 480)
        self.show_stats = True
        self.show_difference = False
        self.sync_display = True
        
        # 窗口名称
        self.main_window = "Camera Comparator"
        self.control_window = "Controls"
        
        # 线程控制
        self.running = False
        self.capture_threads = {}
        
        # 差异检测参数
        self.diff_threshold = 30
        self.reference_camera = device_ids[0] if device_ids else 0

    def open_cameras(self) -> bool:
        """打开所有摄像头"""
        success_count = 0
        
        for device_id in self.device_ids:
            try:
                cap = cv2.VideoCapture(device_id)
                if cap.isOpened():
                    # 设置分辨率
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_size[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_size[1])
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    self.caps[device_id] = cap
                    self.stats[device_id] = CameraStats(device_id)
                    self.frames[device_id] = None
                    self.frame_history[device_id] = deque(maxlen=self.history_size)
                    
                    print(f"摄像头 {device_id} 已打开")
                    success_count += 1
                else:
                    print(f"无法打开摄像头 {device_id}")
                    
            except Exception as e:
                print(f"打开摄像头 {device_id} 时发生错误: {e}")
        
        if success_count == 0:
            print("没有可用的摄像头")
            return False
        
        print(f"成功打开 {success_count}/{len(self.device_ids)} 个摄像头")
        return True

    def close_cameras(self):
        """关闭所有摄像头"""
        self.running = False
        
        # 等待所有线程结束
        for thread in self.capture_threads.values():
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        # 释放所有摄像头
        for cap in self.caps.values():
            cap.release()
        
        cv2.destroyAllWindows()

    def capture_frame_thread(self, device_id: int):
        """单个摄像头的捕获线程"""
        cap = self.caps[device_id]
        stats = self.stats[device_id]
        
        while self.running:
            start_time = time.time()
            
            ret, frame = cap.read()
            if ret:
                # 调整帧大小
                frame = cv2.resize(frame, self.display_size)
                
                # 更新统计信息
                current_time = time.time()
                stats.frame_count += 1
                stats.last_frame_time = current_time
                stats.avg_latency = current_time - start_time
                
                # 计算画质指标
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                stats.brightness_avg = np.mean(gray)
                stats.contrast_score = np.std(gray)
                
                # 更新帧
                self.frames[device_id] = frame.copy()
                
                # 保存到历史记录
                self.frame_history[device_id].append({
                    'frame': frame.copy(),
                    'timestamp': current_time,
                    'stats': {
                        'brightness': stats.brightness_avg,
                        'contrast': stats.contrast_score
                    }
                })
                
                # 计算FPS
                if stats.frame_count > 1:
                    fps_interval = current_time - stats.last_frame_time if stats.last_frame_time > 0 else 0.033
                    if fps_interval > 0:
                        stats.fps = 0.9 * stats.fps + 0.1 * (1.0 / fps_interval)
            else:
                stats.dropped_frames += 1
            
            time.sleep(0.01)  # 避免过度占用CPU

    def start_capture_threads(self):
        """启动所有捕获线程"""
        self.running = True
        
        for device_id in self.caps.keys():
            thread = threading.Thread(
                target=self.capture_frame_thread,
                args=(device_id,),
                daemon=True
            )
            thread.start()
            self.capture_threads[device_id] = thread

    def create_grid_layout(self, frames: Dict[int, np.ndarray]) -> np.ndarray:
        """创建网格布局"""
        if not frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        device_count = len(frames)
        
        # 计算网格尺寸
        if device_count == 1:
            rows, cols = 1, 1
        elif device_count == 2:
            rows, cols = 1, 2
        elif device_count <= 4:
            rows, cols = 2, 2
        elif device_count <= 6:
            rows, cols = 2, 3
        elif device_count <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 4, 4
        
        # 计算每个子窗口大小
        cell_width = self.display_size[0] // cols
        cell_height = self.display_size[1] // rows
        
        # 创建输出图像
        output_width = cols * cell_width
        output_height = rows * cell_height
        result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # 填充子窗口
        for i, (device_id, frame) in enumerate(frames.items()):
            if i >= rows * cols:
                break
            
            row = i // cols
            col = i % cols
            
            # 调整帧大小
            resized_frame = cv2.resize(frame, (cell_width, cell_height))
            
            # 添加设备标识
            stats = self.stats[device_id]
            info_text = f"Camera {device_id}"
            cv2.putText(resized_frame, info_text, (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if self.show_stats:
                # 添加统计信息
                stats_text = f"FPS: {stats.fps:.1f}"
                cv2.putText(resized_frame, stats_text, (5, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                brightness_text = f"Bright: {stats.brightness_avg:.0f}"
                cv2.putText(resized_frame, brightness_text, (5, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 放置到结果图像中
            y1 = row * cell_height
            y2 = y1 + cell_height
            x1 = col * cell_width
            x2 = x1 + cell_width
            
            result[y1:y2, x1:x2] = resized_frame
        
        return result

    def create_horizontal_layout(self, frames: Dict[int, np.ndarray]) -> np.ndarray:
        """创建水平布局"""
        if not frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        frame_list = []
        cell_width = self.display_size[0] // len(frames)
        
        for device_id, frame in frames.items():
            # 调整大小
            resized_frame = cv2.resize(frame, (cell_width, self.display_size[1]))
            
            # 添加标识
            info_text = f"Cam {device_id}"
            cv2.putText(resized_frame, info_text, (5, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            frame_list.append(resized_frame)
        
        return np.hstack(frame_list)

    def create_vertical_layout(self, frames: Dict[int, np.ndarray]) -> np.ndarray:
        """创建垂直布局"""
        if not frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        frame_list = []
        cell_height = self.display_size[1] // len(frames)
        
        for device_id, frame in frames.items():
            # 调整大小
            resized_frame = cv2.resize(frame, (self.display_size[0], cell_height))
            
            # 添加标识
            info_text = f"Camera {device_id}"
            cv2.putText(resized_frame, info_text, (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            frame_list.append(resized_frame)
        
        return np.vstack(frame_list)

    def calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """计算两帧之间的差异"""
        # 转换为灰度图
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 计算差异
        diff = cv2.absdiff(gray1, gray2)
        
        # 阈值处理
        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        
        # 转换回彩色图像以便显示
        diff_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        return diff_colored

    def create_difference_display(self, frames: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
        """创建差异对比显示"""
        if len(frames) < 2 or self.reference_camera not in frames:
            return None
        
        reference_frame = frames[self.reference_camera]
        diff_frames = {}
        
        for device_id, frame in frames.items():
            if device_id != self.reference_camera:
                diff_frame = self.calculate_frame_difference(reference_frame, frame)
                diff_frames[device_id] = diff_frame
        
        if not diff_frames:
            return None
        
        # 使用网格布局显示差异
        return self.create_grid_layout(diff_frames)

    def print_comparison_stats(self):
        """打印对比统计信息"""
        print("\n=== 摄像头对比统计 ===")
        print(f"{'设备ID':<8} {'FPS':<8} {'帧数':<10} {'丢帧':<8} {'亮度':<8} {'对比度':<10} {'延迟(ms)':<10}")
        print("-" * 70)
        
        for device_id, stats in self.stats.items():
            print(f"{device_id:<8} {stats.fps:<8.1f} {stats.frame_count:<10} "
                  f"{stats.dropped_frames:<8} {stats.brightness_avg:<8.0f} "
                  f"{stats.contrast_score:<10.1f} {stats.avg_latency*1000:<10.1f}")

    def save_comparison_image(self, display_frame: np.ndarray, filename: Optional[str] = None):
        """保存对比图像"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_comparison_{timestamp}.jpg"
        
        cv2.imwrite(filename, display_frame)
        print(f"对比图像已保存到: {filename}")

    def run_comparison(self):
        """运行对比界面"""
        if not self.caps:
            print("没有可用的摄像头")
            return
        
        # 启动捕获线程
        self.start_capture_threads()
        
        print("摄像头对比工具已启动")
        print("按键控制:")
        print("  1-3: 切换显示模式 (网格/水平/垂直)")
        print("  d: 切换差异显示")
        print("  s: 切换统计信息显示")
        print("  c: 截图保存对比结果")
        print("  i: 打印统计信息")
        print("  r: 重置统计")
        print("  +/-: 调节差异阈值")
        print("  q: 退出")
        
        try:
            while True:
                # 获取当前帧
                current_frames = {}
                for device_id in self.caps.keys():
                    if self.frames[device_id] is not None:
                        current_frames[device_id] = self.frames[device_id].copy()
                
                if not current_frames:
                    time.sleep(0.1)
                    continue
                
                # 创建显示画面
                if self.show_difference:
                    display_frame = self.create_difference_display(current_frames)
                    if display_frame is None:
                        display_frame = self.create_grid_layout(current_frames)
                else:
                    if self.display_mode == 'grid':
                        display_frame = self.create_grid_layout(current_frames)
                    elif self.display_mode == 'horizontal':
                        display_frame = self.create_horizontal_layout(current_frames)
                    elif self.display_mode == 'vertical':
                        display_frame = self.create_vertical_layout(current_frames)
                    else:
                        display_frame = self.create_grid_layout(current_frames)
                
                # 添加全局信息
                if self.show_stats:
                    info_text = f"Mode: {self.display_mode} | Cameras: {len(current_frames)} | Diff Threshold: {self.diff_threshold}"
                    cv2.putText(display_frame, info_text, (10, display_frame.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # 显示
                cv2.imshow(self.main_window, display_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    self.display_mode = 'grid'
                    print("切换到网格模式")
                elif key == ord('2'):
                    self.display_mode = 'horizontal'
                    print("切换到水平模式")
                elif key == ord('3'):
                    self.display_mode = 'vertical'
                    print("切换到垂直模式")
                elif key == ord('d'):
                    self.show_difference = not self.show_difference
                    status = "开启" if self.show_difference else "关闭"
                    print(f"差异显示已{status}")
                elif key == ord('s'):
                    self.show_stats = not self.show_stats
                    status = "开启" if self.show_stats else "关闭"
                    print(f"统计信息显示已{status}")
                elif key == ord('c'):
                    self.save_comparison_image(display_frame)
                elif key == ord('i'):
                    self.print_comparison_stats()
                elif key == ord('r'):
                    self._reset_stats()
                    print("统计信息已重置")
                elif key == ord('+') or key == ord('='):
                    self.diff_threshold = min(255, self.diff_threshold + 5)
                    print(f"差异阈值: {self.diff_threshold}")
                elif key == ord('-'):
                    self.diff_threshold = max(5, self.diff_threshold - 5)
                    print(f"差异阈值: {self.diff_threshold}")
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            self.close_cameras()

    def _reset_stats(self):
        """重置统计信息"""
        for stats in self.stats.values():
            stats.frame_count = 0
            stats.dropped_frames = 0
            stats.fps = 0.0
            stats.avg_latency = 0.0


def main():
    parser = argparse.ArgumentParser(description="摄像头画面对比工具")
    parser.add_argument('devices', nargs='+', type=int, help='摄像头设备ID列表')
    parser.add_argument('--width', type=int, default=640, help='显示宽度，默认640')
    parser.add_argument('--height', type=int, default=480, help='显示高度，默认480')
    parser.add_argument('--mode', type=str, default='grid', 
                       choices=['grid', 'horizontal', 'vertical'],
                       help='显示模式，默认grid')
    parser.add_argument('--diff-threshold', type=int, default=30, help='差异检测阈值，默认30')
    parser.add_argument('--reference', type=int, help='参考摄像头ID，用于差异对比')
    parser.add_argument('--no-stats', action='store_true', help='不显示统计信息')
    
    args = parser.parse_args()

    if len(args.devices) < 2:
        print("需要至少2个摄像头进行对比")
        sys.exit(1)

    comparator = CameraComparator(args.devices)
    
    # 设置参数
    comparator.display_size = (args.width, args.height)
    comparator.display_mode = args.mode
    comparator.diff_threshold = args.diff_threshold
    comparator.show_stats = not args.no_stats
    
    if args.reference is not None:
        comparator.reference_camera = args.reference
    
    try:
        if not comparator.open_cameras():
            sys.exit(1)
        
        comparator.run_comparison()
        
    except Exception as e:
        print(f"发生错误：{e}")
    finally:
        comparator.close_cameras()


if __name__ == "__main__":
    main()