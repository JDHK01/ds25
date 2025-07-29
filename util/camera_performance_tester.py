#!/usr/bin/env python3
"""
摄像头性能测试工具
测试摄像头的帧率、延迟、资源使用等性能指标
"""

import cv2
import numpy as np
import argparse
import sys
import time
import threading
import psutil
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque

@dataclass
class PerformanceMetrics:
    """性能指标"""
    device_id: int
    resolution: Tuple[int, int]
    target_fps: float
    actual_fps: float
    frame_count: int
    dropped_frames: int
    min_frame_time: float
    max_frame_time: float
    avg_frame_time: float
    frame_time_std: float
    cpu_usage: float
    memory_usage_mb: float
    test_duration: float
    timestamp: str

class CameraPerformanceTester:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.cap = None
        
        # 测试参数
        self.test_resolutions = [
            (320, 240),   # QVGA
            (640, 480),   # VGA
            (800, 600),   # SVGA
            (1280, 720),  # HD
            (1920, 1080), # Full HD
        ]
        
        self.test_fps_values = [15, 30, 60]
        
        # 性能数据
        self.frame_times = deque(maxlen=1000)
        self.cpu_samples = deque(maxlen=100)
        self.memory_samples = deque(maxlen=100)
        
        # 测试控制
        self.test_running = False
        self.start_time = 0
        self.frame_count = 0
        self.dropped_frames = 0
        
        # 资源监控线程
        self.monitor_thread = None
        self.process = psutil.Process()

    def open_camera(self) -> bool:
        """打开摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                print(f"无法打开摄像头 {self.device_id}")
                return False
            
            print(f"摄像头 {self.device_id} 已打开")
            return True
            
        except Exception as e:
            print(f"打开摄像头时发生错误：{e}")
            return False

    def close_camera(self):
        """关闭摄像头"""
        self.test_running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def set_camera_params(self, width: int, height: int, fps: float):
        """设置摄像头参数"""
        if not self.cap:
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # 验证设置
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        return (actual_width == width and actual_height == height)

    def monitor_resources(self):
        """监控系统资源使用"""
        while self.test_running:
            try:
                # CPU使用率
                cpu_percent = self.process.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                
                # 内存使用
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.memory_samples.append(memory_mb)
                
                time.sleep(0.1)  # 每100ms采样一次
                
            except Exception as e:
                print(f"资源监控错误: {e}")
                break

    def test_single_configuration(self, width: int, height: int, fps: float, 
                                 duration: float = 10.0, show_preview: bool = False) -> PerformanceMetrics:
        """测试单个配置的性能"""
        print(f"测试配置: {width}x{height} @ {fps} fps，时长 {duration} 秒")
        
        # 设置参数
        if not self.set_camera_params(width, height, fps):
            print(f"警告: 无法设置到指定分辨率 {width}x{height}")
        
        # 获取实际参数
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 重置统计数据
        self.frame_times.clear()
        self.cpu_samples.clear()
        self.memory_samples.clear()
        self.frame_count = 0
        self.dropped_frames = 0
        
        # 开始资源监控
        self.test_running = True
        self.monitor_thread = threading.Thread(target=self.monitor_resources, daemon=True)
        self.monitor_thread.start()
        
        # 开始测试
        self.start_time = time.time()
        window_name = f"Performance Test {width}x{height}@{fps}fps" if show_preview else None
        
        if show_preview:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        last_frame_time = time.time()
        
        while (time.time() - self.start_time) < duration:
            frame_start = time.time()
            
            ret, frame = self.cap.read()
            
            if ret:
                self.frame_count += 1
                
                # 记录帧时间
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                
                # 显示预览（如果启用）
                if show_preview and frame is not None:
                    # 添加性能信息到画面
                    current_fps = 1.0 / (time.time() - last_frame_time) if (time.time() - last_frame_time) > 0 else 0
                    info_text = f"FPS: {current_fps:.1f} | Frames: {self.frame_count}"
                    cv2.putText(frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow(window_name, frame)
                    
                    # 检查退出按键
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                last_frame_time = time.time()
                
            else:
                self.dropped_frames += 1
        
        # 停止测试
        self.test_running = False
        if show_preview:
            cv2.destroyWindow(window_name)
        
        # 计算性能指标
        test_duration = time.time() - self.start_time
        actual_fps = self.frame_count / test_duration if test_duration > 0 else 0
        
        frame_times_array = np.array(list(self.frame_times))
        min_frame_time = np.min(frame_times_array) if len(frame_times_array) > 0 else 0
        max_frame_time = np.max(frame_times_array) if len(frame_times_array) > 0 else 0
        avg_frame_time = np.mean(frame_times_array) if len(frame_times_array) > 0 else 0
        frame_time_std = np.std(frame_times_array) if len(frame_times_array) > 0 else 0
        
        avg_cpu = np.mean(list(self.cpu_samples)) if self.cpu_samples else 0
        avg_memory = np.mean(list(self.memory_samples)) if self.memory_samples else 0
        
        return PerformanceMetrics(
            device_id=self.device_id,
            resolution=(actual_width, actual_height),
            target_fps=fps,
            actual_fps=actual_fps,
            frame_count=self.frame_count,
            dropped_frames=self.dropped_frames,
            min_frame_time=min_frame_time * 1000,  # 转换为毫秒
            max_frame_time=max_frame_time * 1000,
            avg_frame_time=avg_frame_time * 1000,
            frame_time_std=frame_time_std * 1000,
            cpu_usage=avg_cpu,
            memory_usage_mb=avg_memory,
            test_duration=test_duration,
            timestamp=datetime.now().isoformat()
        )

    def run_comprehensive_test(self, duration_per_test: float = 10.0, 
                              show_preview: bool = False) -> List[PerformanceMetrics]:
        """运行全面的性能测试"""
        if not self.cap:
            print("摄像头未打开")
            return []
        
        results = []
        total_tests = len(self.test_resolutions) * len(self.test_fps_values)
        current_test = 0
        
        print(f"开始全面性能测试，共 {total_tests} 个配置")
        print("=" * 60)
        
        for resolution in self.test_resolutions:
            for fps in self.test_fps_values:
                current_test += 1
                print(f"\n进度 {current_test}/{total_tests}")
                
                try:
                    metrics = self.test_single_configuration(
                        resolution[0], resolution[1], fps, 
                        duration_per_test, show_preview
                    )
                    results.append(metrics)
                    
                    # 打印简要结果
                    print(f"结果: {metrics.actual_fps:.1f} fps (目标 {fps}), "
                          f"CPU: {metrics.cpu_usage:.1f}%, "
                          f"内存: {metrics.memory_usage_mb:.1f} MB")
                    
                except Exception as e:
                    print(f"测试配置 {resolution}@{fps} 失败: {e}")
                
                # 测试间短暂休息
                if current_test < total_tests:
                    time.sleep(1)
        
        return results

    def test_stress_performance(self, duration: float = 60.0) -> Dict:
        """压力测试"""
        print(f"开始压力测试，持续 {duration} 秒...")
        
        # 使用最高分辨率和帧率
        max_resolution = max(self.test_resolutions, key=lambda x: x[0] * x[1])
        max_fps = max(self.test_fps_values)
        
        self.set_camera_params(max_resolution[0], max_resolution[1], max_fps)
        
        # 重置统计
        self.frame_times.clear()
        self.cpu_samples.clear()
        self.memory_samples.clear()
        self.frame_count = 0
        self.dropped_frames = 0
        
        # 开始监控
        self.test_running = True
        self.monitor_thread = threading.Thread(target=self.monitor_resources, daemon=True)
        self.monitor_thread.start()
        
        self.start_time = time.time()
        fps_samples = []
        
        print("压力测试进行中...")
        
        while (time.time() - self.start_time) < duration:
            frame_start = time.time()
            
            ret, frame = self.cap.read()
            
            if ret:
                self.frame_count += 1
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                
                # 每秒计算一次FPS
                if len(self.frame_times) >= 30:
                    recent_times = list(self.frame_times)[-30:]
                    current_fps = 30 / sum(recent_times)
                    fps_samples.append(current_fps)
                
                # 定期输出进度
                elapsed = time.time() - self.start_time
                if self.frame_count % 300 == 0:  # 每10秒左右
                    progress = elapsed / duration * 100
                    current_fps = self.frame_count / elapsed
                    print(f"进度: {progress:.1f}%, 当前FPS: {current_fps:.1f}")
                
            else:
                self.dropped_frames += 1
        
        self.test_running = False
        
        # 分析结果
        test_duration = time.time() - self.start_time
        avg_fps = self.frame_count / test_duration
        
        cpu_usage = list(self.cpu_samples)
        memory_usage = list(self.memory_samples)
        
        results = {
            'duration': test_duration,
            'total_frames': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'average_fps': avg_fps,
            'fps_stability': np.std(fps_samples) if fps_samples else 0,
            'min_fps': np.min(fps_samples) if fps_samples else 0,
            'max_fps': np.max(fps_samples) if fps_samples else 0,
            'cpu_usage': {
                'average': np.mean(cpu_usage) if cpu_usage else 0,
                'max': np.max(cpu_usage) if cpu_usage else 0,
                'min': np.min(cpu_usage) if cpu_usage else 0
            },
            'memory_usage_mb': {
                'average': np.mean(memory_usage) if memory_usage else 0,
                'max': np.max(memory_usage) if memory_usage else 0,
                'min': np.min(memory_usage) if memory_usage else 0
            }
        }
        
        return results

    def print_results(self, results: List[PerformanceMetrics]):
        """打印测试结果"""
        print("\n" + "=" * 80)
        print("性能测试结果")
        print("=" * 80)
        
        print(f"{'分辨率':<12} {'目标FPS':<8} {'实际FPS':<8} {'帧时间(ms)':<12} "
              f"{'CPU%':<6} {'内存(MB)':<10} {'丢帧':<6}")
        print("-" * 80)
        
        for result in results:
            resolution_str = f"{result.resolution[0]}x{result.resolution[1]}"
            print(f"{resolution_str:<12} {result.target_fps:<8.1f} {result.actual_fps:<8.1f} "
                  f"{result.avg_frame_time:<12.2f} {result.cpu_usage:<6.1f} "
                  f"{result.memory_usage_mb:<10.1f} {result.dropped_frames:<6}")

    def print_stress_results(self, results: Dict):
        """打印压力测试结果"""
        print("\n" + "=" * 60)
        print("压力测试结果")
        print("=" * 60)
        
        print(f"测试时长: {results['duration']:.1f} 秒")
        print(f"总帧数: {results['total_frames']}")
        print(f"丢帧数: {results['dropped_frames']}")
        print(f"平均FPS: {results['average_fps']:.2f}")
        print(f"FPS稳定性(标准差): {results['fps_stability']:.2f}")
        print(f"FPS范围: {results['min_fps']:.1f} - {results['max_fps']:.1f}")
        
        print(f"\nCPU使用率:")
        print(f"  平均: {results['cpu_usage']['average']:.1f}%")
        print(f"  最大: {results['cpu_usage']['max']:.1f}%")
        print(f"  最小: {results['cpu_usage']['min']:.1f}%")
        
        print(f"\n内存使用:")
        print(f"  平均: {results['memory_usage_mb']['average']:.1f} MB")
        print(f"  最大: {results['memory_usage_mb']['max']:.1f} MB")
        print(f"  最小: {results['memory_usage_mb']['min']:.1f} MB")

    def save_results(self, results: List[PerformanceMetrics], filename: Optional[str] = None):
        """保存测试结果到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_performance_{self.device_id}_{timestamp}.json"
        
        # 转换为字典格式
        results_dict = {
            'device_id': self.device_id,
            'test_timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'platform': sys.platform
            },
            'results': [asdict(result) for result in results]
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            print(f"测试结果已保存到: {filename}")
        except Exception as e:
            print(f"保存结果失败: {e}")

    def test_latency(self, duration: float = 10.0) -> Dict:
        """测试摄像头延迟"""
        print(f"测试摄像头延迟，持续 {duration} 秒...")
        
        # 使用中等分辨率以平衡精度和性能
        self.set_camera_params(640, 480, 30)
        
        latencies = []
        self.start_time = time.time()
        
        while (time.time() - self.start_time) < duration:
            request_time = time.time()
            ret, frame = self.cap.read()
            response_time = time.time()
            
            if ret:
                latency = (response_time - request_time) * 1000  # 转换为毫秒
                latencies.append(latency)
            
            time.sleep(0.1)  # 100ms间隔测试
        
        if latencies:
            return {
                'avg_latency_ms': np.mean(latencies),
                'min_latency_ms': np.min(latencies),
                'max_latency_ms': np.max(latencies),
                'latency_std_ms': np.std(latencies),
                'samples': len(latencies)
            }
        else:
            return {'error': '无法获取延迟数据'}


def main():
    parser = argparse.ArgumentParser(description="摄像头性能测试工具")
    parser.add_argument('--device', type=int, default=0, help='摄像头设备编号，默认0')
    parser.add_argument('--duration', type=float, default=10.0, help='每项测试持续时间（秒），默认10')
    parser.add_argument('--stress', action='store_true', help='运行压力测试')
    parser.add_argument('--stress-duration', type=float, default=60.0, help='压力测试持续时间（秒），默认60')
    parser.add_argument('--latency', action='store_true', help='测试延迟')
    parser.add_argument('--preview', action='store_true', help='显示测试预览')
    parser.add_argument('--save', type=str, help='保存结果到指定文件')
    parser.add_argument('--single', nargs=3, type=float, metavar=('WIDTH', 'HEIGHT', 'FPS'),
                       help='测试单个配置: 宽度 高度 帧率')
    
    args = parser.parse_args()

    tester = CameraPerformanceTester(args.device)
    
    try:
        if not tester.open_camera():
            sys.exit(1)
        
        if args.single:
            # 测试单个配置
            width, height, fps = int(args.single[0]), int(args.single[1]), args.single[2]
            print(f"测试单个配置: {width}x{height}@{fps}fps")
            
            result = tester.test_single_configuration(width, height, fps, 
                                                    args.duration, args.preview)
            tester.print_results([result])
            
            if args.save:
                tester.save_results([result], args.save)
        
        elif args.stress:
            # 压力测试
            results = tester.test_stress_performance(args.stress_duration)
            tester.print_stress_results(results)
        
        elif args.latency:
            # 延迟测试
            results = tester.test_latency(args.duration)
            print("\n延迟测试结果:")
            if 'error' in results:
                print(f"错误: {results['error']}")
            else:
                print(f"平均延迟: {results['avg_latency_ms']:.2f} ms")
                print(f"最小延迟: {results['min_latency_ms']:.2f} ms")
                print(f"最大延迟: {results['max_latency_ms']:.2f} ms")
                print(f"延迟标准差: {results['latency_std_ms']:.2f} ms")
                print(f"采样数: {results['samples']}")
        
        else:
            # 全面测试
            results = tester.run_comprehensive_test(args.duration, args.preview)
            tester.print_results(results)
            
            if args.save:
                tester.save_results(results, args.save)
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"发生错误：{e}")
    finally:
        tester.close_camera()


if __name__ == "__main__":
    main()