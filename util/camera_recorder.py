#!/usr/bin/env python3
"""
摄像头录制工具
支持多格式录制、实时预览、暂停恢复等功能
"""

import cv2
import argparse
import os
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Tuple

class CameraRecorder:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.cap = None
        self.video_writer = None
        self.is_recording = False
        self.is_paused = False
        self.start_time = None
        self.pause_duration = 0
        self.total_frames = 0
        
        # 录制参数
        self.output_path = None
        self.fps = 30.0
        self.width = 1280
        self.height = 720
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        # 窗口名称
        self.window_name = f"Camera {device_id} Recorder"
        
        # 支持的视频格式
        self.supported_formats = {
            'avi': cv2.VideoWriter_fourcc(*'XVID'),
            'mp4': cv2.VideoWriter_fourcc(*'mp4v'),
            'mov': cv2.VideoWriter_fourcc(*'mp4v'),
            'mkv': cv2.VideoWriter_fourcc(*'XVID'),
            'wmv': cv2.VideoWriter_fourcc(*'WMV1'),
        }

    def open_camera(self) -> bool:
        """打开摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                print(f"无法打开摄像头 {self.device_id}")
                return False
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 获取实际设置的参数
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if actual_fps > 0:
                self.fps = actual_fps
            
            print(f"摄像头 {self.device_id} 已打开")
            print(f"分辨率: {self.width}x{self.height}")
            print(f"帧率: {self.fps:.1f} fps")
            
            return True
            
        except Exception as e:
            print(f"打开摄像头时发生错误：{e}")
            return False

    def close_camera(self):
        """关闭摄像头"""
        self.stop_recording()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def set_resolution(self, width: int, height: int):
        """设置录制分辨率"""
        self.width = width
        self.height = height
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def set_fps(self, fps: float):
        """设置录制帧率"""
        self.fps = fps
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def start_recording(self, output_path: str, format_ext: str = 'avi') -> bool:
        """开始录制"""
        if self.is_recording:
            print("已在录制中")
            return False
        
        if not self.cap:
            print("摄像头未打开")
            return False
        
        # 确定输出文件路径
        if not output_path.endswith(f'.{format_ext}'):
            output_path = f"{output_path}.{format_ext}"
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 获取编码器
        fourcc = self.supported_formats.get(format_ext, self.fourcc)
        
        try:
            # 创建视频写入器
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.fps, (self.width, self.height)
            )
            
            if not self.video_writer.isOpened():
                print(f"无法创建视频文件: {output_path}")
                return False
            
            self.output_path = output_path
            self.is_recording = True
            self.is_paused = False
            self.start_time = time.time()
            self.pause_duration = 0
            self.total_frames = 0
            
            print(f"开始录制到: {output_path}")
            return True
            
        except Exception as e:
            print(f"开始录制时发生错误: {e}")
            return False

    def stop_recording(self):
        """停止录制"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.is_paused = False
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        if self.output_path:
            file_size = os.path.getsize(self.output_path) / (1024 * 1024)  # MB
            duration = time.time() - self.start_time - self.pause_duration
            print(f"录制完成: {self.output_path}")
            print(f"文件大小: {file_size:.2f} MB")
            print(f"录制时长: {duration:.1f} 秒")
            print(f"总帧数: {self.total_frames}")

    def pause_recording(self):
        """暂停录制"""
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            self.pause_start_time = time.time()
            print("录制已暂停")

    def resume_recording(self):
        """恢复录制"""
        if self.is_recording and self.is_paused:
            self.is_paused = False
            self.pause_duration += time.time() - self.pause_start_time
            print("录制已恢复")

    def get_recording_info(self) -> dict:
        """获取录制信息"""
        if not self.is_recording:
            return {}
        
        current_time = time.time()
        total_duration = current_time - self.start_time
        effective_duration = total_duration - self.pause_duration
        
        if self.is_paused:
            effective_duration -= (current_time - self.pause_start_time)
        
        return {
            'is_recording': self.is_recording,
            'is_paused': self.is_paused,
            'total_duration': total_duration,
            'effective_duration': effective_duration,
            'total_frames': self.total_frames,
            'output_path': self.output_path,
            'file_size_mb': os.path.getsize(self.output_path) / (1024 * 1024) if self.output_path and os.path.exists(self.output_path) else 0
        }

    def draw_recording_info(self, frame):
        """在画面上绘制录制信息"""
        if not self.is_recording:
            return
        
        info = self.get_recording_info()
        
        # 背景矩形
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 录制状态
        status_text = "PAUSED" if self.is_paused else "RECORDING"
        status_color = (0, 255, 255) if self.is_paused else (0, 0, 255)
        cv2.putText(frame, status_text, (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
        # 时间信息
        duration_str = self.format_duration(info['effective_duration'])
        cv2.putText(frame, f"Duration: {duration_str}", (15, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 帧数和文件信息
        cv2.putText(frame, f"Frames: {info['total_frames']}", (15, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Size: {info['file_size_mb']:.1f} MB", (15, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def format_duration(self, seconds: float) -> str:
        """格式化时长显示"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def run_preview_mode(self, show_info: bool = True):
        """运行预览模式"""
        if not self.cap:
            print("摄像头未打开")
            return
        
        print("预览模式 - 按键控制:")
        print("  SPACE: 开始/停止录制")
        print("  p: 暂停/恢复录制")
        print("  s: 截图")
        print("  i: 切换信息显示")
        print("  q: 退出")
        
        screenshot_counter = 1
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 录制帧
            if self.is_recording and not self.is_paused and self.video_writer:
                self.video_writer.write(frame)
                self.total_frames += 1
            
            # 绘制录制信息
            if show_info:
                self.draw_recording_info(frame)
            
            # 显示画面
            cv2.imshow(self.window_name, frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键
                if self.is_recording:
                    self.stop_recording()
                else:
                    # 生成默认文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"recording_{timestamp}"
                    self.start_recording(filename)
            elif key == ord('p'):
                if self.is_recording:
                    if self.is_paused:
                        self.resume_recording()
                    else:
                        self.pause_recording()
            elif key == ord('s'):
                # 截图
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"screenshot_{timestamp}_{screenshot_counter:03d}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"截图保存到: {screenshot_path}")
                screenshot_counter += 1
            elif key == ord('i'):
                show_info = not show_info

    def run_timed_recording(self, output_path: str, duration_seconds: int, 
                           format_ext: str = 'avi', countdown: int = 3):
        """定时录制"""
        if not self.cap:
            print("摄像头未打开")
            return
        
        # 倒计时
        print(f"将在 {countdown} 秒后开始录制...")
        for i in range(countdown, 0, -1):
            print(f"倒计时: {i}")
            time.sleep(1)
        
        # 开始录制
        if not self.start_recording(output_path, format_ext):
            return
        
        print(f"定时录制 {duration_seconds} 秒...")
        
        # 录制循环
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time and self.is_recording:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 写入帧
            if self.video_writer:
                self.video_writer.write(frame)
                self.total_frames += 1
            
            # 显示进度
            remaining = end_time - time.time()
            progress = (duration_seconds - remaining) / duration_seconds * 100
            
            # 在画面上显示进度
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, "TIMED RECORDING", (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Progress: {progress:.1f}%", (15, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow(self.window_name, frame)
            
            # 检查中断
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("录制被用户中断")
                break
        
        # 停止录制
        self.stop_recording()

    def batch_record(self, output_dir: str, count: int, duration_each: int, 
                     interval: int = 5, format_ext: str = 'avi'):
        """批量录制"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"批量录制 {count} 个视频，每个 {duration_each} 秒，间隔 {interval} 秒")
        
        for i in range(count):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"batch_{i+1:03d}_{timestamp}")
            
            print(f"\n=== 录制第 {i+1}/{count} 个视频 ===")
            self.run_timed_recording(output_path, duration_each, format_ext, countdown=1)
            
            if i < count - 1:  # 不是最后一个
                print(f"等待 {interval} 秒后录制下一个...")
                time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="摄像头录制工具")
    parser.add_argument('--device', type=int, default=0, help='摄像头设备编号，默认0')
    parser.add_argument('--output', '-o', type=str, help='输出文件路径')
    parser.add_argument('--format', type=str, default='avi', 
                       choices=['avi', 'mp4', 'mov', 'mkv', 'wmv'], 
                       help='视频格式，默认avi')
    parser.add_argument('--fps', type=float, default=30.0, help='帧率，默认30')
    parser.add_argument('--width', type=int, default=1280, help='宽度，默认1280')
    parser.add_argument('--height', type=int, default=720, help='高度，默认720')
    parser.add_argument('--duration', type=int, help='录制时长（秒）')
    parser.add_argument('--countdown', type=int, default=3, help='开始录制前倒计时（秒）')
    parser.add_argument('--batch', type=int, help='批量录制数量')
    parser.add_argument('--batch-duration', type=int, default=60, help='批量录制每个视频时长')
    parser.add_argument('--batch-interval', type=int, default=5, help='批量录制间隔时间')
    parser.add_argument('--no-preview', action='store_true', help='不显示预览（仅用于定时录制）')
    
    args = parser.parse_args()

    recorder = CameraRecorder(args.device)
    
    try:
        # 设置参数
        recorder.set_fps(args.fps)
        recorder.set_resolution(args.width, args.height)
        
        if not recorder.open_camera():
            sys.exit(1)
        
        # 根据参数选择模式
        if args.batch:
            # 批量录制模式
            output_dir = args.output or f"batch_recordings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            recorder.batch_record(output_dir, args.batch, args.batch_duration, 
                                args.batch_interval, args.format)
        
        elif args.duration:
            # 定时录制模式
            output_path = args.output or f"timed_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if not args.no_preview:
                recorder.run_timed_recording(output_path, args.duration, args.format, args.countdown)
            else:
                # 无预览定时录制
                if recorder.start_recording(output_path, args.format):
                    print(f"录制 {args.duration} 秒...")
                    time.sleep(args.duration)
                    recorder.stop_recording()
        
        else:
            # 交互式预览模式
            if args.output:
                # 自动开始录制
                recorder.start_recording(args.output, args.format)
            
            recorder.run_preview_mode()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误：{e}")
    finally:
        recorder.close_camera()


if __name__ == "__main__":
    main()