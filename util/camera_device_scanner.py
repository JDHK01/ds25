#!/usr/bin/env python3
"""
摄像头多设备扫描工具
自动扫描系统中所有可用的摄像头设备并提供管理功能
"""

import cv2
import argparse
import sys
import threading
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class CameraDevice:
    """摄像头设备信息"""
    device_id: int
    name: str
    is_available: bool
    resolution: Tuple[int, int]
    fps: float
    backend: str
    
    def __str__(self):
        status = "可用" if self.is_available else "不可用"
        return f"设备 {self.device_id}: {self.name} [{status}] {self.resolution[0]}x{self.resolution[1]} @{self.fps:.1f}fps"

class CameraScanner:
    def __init__(self, max_devices: int = 10):
        self.max_devices = max_devices
        self.devices: List[CameraDevice] = []
        self.preview_threads = {}
        self.preview_running = {}

    def scan_devices(self, verbose: bool = False) -> List[CameraDevice]:
        """扫描所有可用的摄像头设备"""
        self.devices.clear()
        
        print(f"正在扫描摄像头设备 (0-{self.max_devices-1})...")
        
        for device_id in range(self.max_devices):
            if verbose:
                print(f"检测设备 {device_id}...", end=' ')
            
            device_info = self._probe_device(device_id)
            if device_info:
                self.devices.append(device_info)
                if verbose:
                    print("✓ 找到")
            else:
                if verbose:
                    print("✗ 无设备")
        
        print(f"扫描完成，找到 {len(self.devices)} 个可用设备")
        return self.devices

    def _probe_device(self, device_id: int) -> Optional[CameraDevice]:
        """探测单个设备"""
        cap = None
        try:
            # 尝试打开设备
            cap = cv2.VideoCapture(device_id)
            
            # 检查是否成功打开
            if not cap.isOpened():
                return None
            
            # 尝试读取一帧来确认设备真正可用
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
            
            # 获取设备属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend_name = cap.getBackendName()
            
            # 获取设备名称（简化版）
            device_name = f"Camera {device_id}"
            
            return CameraDevice(
                device_id=device_id,
                name=device_name,
                is_available=True,
                resolution=(width, height),
                fps=fps if fps > 0 else 30.0,
                backend=backend_name
            )
            
        except Exception as e:
            return None
        finally:
            if cap:
                cap.release()

    def list_devices(self, detailed: bool = False):
        """列出所有设备"""
        if not self.devices:
            print("未找到任何摄像头设备")
            return
        
        print(f"\n发现 {len(self.devices)} 个摄像头设备:")
        print("=" * 60)
        
        for device in self.devices:
            print(f"{device}")
            
            if detailed:
                # 获取更详细的信息
                self._print_detailed_info(device)
                print("-" * 40)

    def _print_detailed_info(self, device: CameraDevice):
        """打印设备详细信息"""
        cap = None
        try:
            cap = cv2.VideoCapture(device.device_id)
            if cap.isOpened():
                print(f"  后端: {device.backend}")
                print(f"  分辨率: {device.resolution[0]}x{device.resolution[1]}")
                print(f"  帧率: {device.fps:.1f} fps")
                
                # 尝试获取更多属性
                brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
                contrast = cap.get(cv2.CAP_PROP_CONTRAST)
                saturation = cap.get(cv2.CAP_PROP_SATURATION)
                
                if brightness != -1:
                    print(f"  亮度: {brightness:.2f}")
                if contrast != -1:
                    print(f"  对比度: {contrast:.2f}")
                if saturation != -1:
                    print(f"  饱和度: {saturation:.2f}")
                    
        except Exception as e:
            print(f"  获取详细信息失败: {e}")
        finally:
            if cap:
                cap.release()

    def preview_device(self, device_id: int):
        """预览单个设备"""
        device = self._find_device(device_id)
        if not device:
            print(f"设备 {device_id} 不可用")
            return
        
        cap = None
        try:
            cap = cv2.VideoCapture(device_id)
            if not cap.isOpened():
                print(f"无法打开设备 {device_id}")
                return
            
            window_name = f"Camera {device_id} Preview"
            print(f"预览设备 {device_id} - 按 'q' 退出")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取视频帧")
                    break
                
                # 在画面上显示设备信息
                info_text = f"Device {device_id}: {device.resolution[0]}x{device.resolution[1]} @{device.fps:.1f}fps"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(window_name, frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"预览设备 {device_id} 时发生错误: {e}")
        finally:
            if cap:
                cap.release()
            cv2.destroyWindow(window_name)

    def preview_all_devices(self):
        """同时预览所有设备"""
        if not self.devices:
            print("没有可用设备")
            return
        
        print(f"同时预览 {len(self.devices)} 个设备 - 按 'q' 退出")
        
        # 为每个设备启动预览线程
        for device in self.devices:
            self.preview_running[device.device_id] = True
            thread = threading.Thread(
                target=self._preview_thread,
                args=(device,),
                daemon=True
            )
            thread.start()
            self.preview_threads[device.device_id] = thread
        
        # 等待用户输入
        try:
            input("按 Enter 键停止所有预览...")
        except KeyboardInterrupt:
            pass
        
        # 停止所有预览
        self._stop_all_previews()

    def _preview_thread(self, device: CameraDevice):
        """单个设备的预览线程"""
        cap = None
        try:
            cap = cv2.VideoCapture(device.device_id)
            if not cap.isOpened():
                print(f"线程中无法打开设备 {device.device_id}")
                return
            
            window_name = f"Device {device.device_id}"
            
            while self.preview_running.get(device.device_id, False):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 调整窗口大小以便同时显示多个
                display_frame = cv2.resize(frame, (400, 300))
                
                # 添加设备信息
                info_text = f"Device {device.device_id}"
                cv2.putText(display_frame, info_text, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow(window_name, display_frame)
                
                # 检查是否有退出按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self._stop_all_previews()
                    break
                    
                time.sleep(0.03)  # 约30fps
                
        except Exception as e:
            print(f"设备 {device.device_id} 预览线程错误: {e}")
        finally:
            if cap:
                cap.release()
            cv2.destroyWindow(window_name)

    def _stop_all_previews(self):
        """停止所有预览"""
        for device_id in self.preview_running:
            self.preview_running[device_id] = False
        
        # 等待所有线程结束
        for thread in self.preview_threads.values():
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        cv2.destroyAllWindows()
        self.preview_threads.clear()
        self.preview_running.clear()

    def compare_devices(self, device_ids: List[int]):
        """比较多个设备的画质和延迟"""
        available_devices = [d for d in self.devices if d.device_id in device_ids]
        
        if len(available_devices) < 2:
            print("需要至少2个设备进行比较")
            return
        
        print(f"比较设备: {[d.device_id for d in available_devices]}")
        
        caps = {}
        windows = {}
        
        try:
            # 打开所有设备
            for device in available_devices:
                cap = cv2.VideoCapture(device.device_id)
                if cap.isOpened():
                    caps[device.device_id] = cap
                    windows[device.device_id] = f"Compare - Device {device.device_id}"
            
            if not caps:
                print("无法打开任何设备进行比较")
                return
            
            print("设备比较模式 - 按 'q' 退出")
            
            while True:
                for device_id, cap in caps.items():
                    ret, frame = cap.read()
                    if ret:
                        # 调整到统一大小便于比较
                        display_frame = cv2.resize(frame, (640, 480))
                        
                        # 添加设备标识和时间戳
                        timestamp = time.time()
                        info_text = f"Device {device_id} - {timestamp:.2f}"
                        cv2.putText(display_frame, info_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.imshow(windows[device_id], display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"设备比较时发生错误: {e}")
        finally:
            for cap in caps.values():
                cap.release()
            cv2.destroyAllWindows()

    def _find_device(self, device_id: int) -> Optional[CameraDevice]:
        """查找设备"""
        for device in self.devices:
            if device.device_id == device_id:
                return device
        return None

    def interactive_mode(self):
        """交互模式"""
        print("\n=== 摄像头设备管理器 ===")
        print("输入命令 (输入 'help' 查看帮助):")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'help':
                    self._show_help()
                elif command == 'scan':
                    self.scan_devices(verbose=True)
                elif command == 'list':
                    self.list_devices()
                elif command == 'detail':
                    self.list_devices(detailed=True)
                elif command.startswith('preview '):
                    try:
                        device_id = int(command.split()[1])
                        self.preview_device(device_id)
                    except (IndexError, ValueError):
                        print("用法: preview <设备ID>")
                elif command == 'preview_all':
                    self.preview_all_devices()
                elif command.startswith('compare '):
                    try:
                        device_ids = [int(x) for x in command.split()[1:]]
                        self.compare_devices(device_ids)
                    except ValueError:
                        print("用法: compare <设备ID1> <设备ID2> ...")
                elif command in ['quit', 'exit', 'q']:
                    break
                else:
                    print("未知命令，输入 'help' 查看帮助")
                    
            except KeyboardInterrupt:
                print("\n程序被中断")
                break
            except Exception as e:
                print(f"命令执行错误: {e}")

    def _show_help(self):
        """显示帮助信息"""
        print("""
可用命令:
  scan              - 重新扫描设备
  list              - 列出所有设备
  detail            - 列出设备详细信息
  preview <ID>      - 预览指定设备
  preview_all       - 同时预览所有设备
  compare <ID1> <ID2>... - 比较多个设备
  help              - 显示此帮助
  quit/exit/q       - 退出程序
        """)


def main():
    parser = argparse.ArgumentParser(description="摄像头多设备扫描工具")
    parser.add_argument('--max-devices', type=int, default=10, help='最大扫描设备数，默认10')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--list', action='store_true', help='扫描并列出设备后退出')
    parser.add_argument('--detail', action='store_true', help='显示设备详细信息')
    parser.add_argument('--preview', type=int, metavar='DEVICE_ID', help='预览指定设备')
    parser.add_argument('--preview-all', action='store_true', help='同时预览所有设备')
    parser.add_argument('--compare', nargs='+', type=int, metavar='DEVICE_ID', help='比较多个设备')
    parser.add_argument('--interactive', '-i', action='store_true', help='进入交互模式')
    
    args = parser.parse_args()

    scanner = CameraScanner(max_devices=args.max_devices)
    
    try:
        # 首次扫描
        scanner.scan_devices(verbose=args.verbose)
        
        if args.list:
            scanner.list_devices()
        elif args.detail:
            scanner.list_devices(detailed=True)
        elif args.preview is not None:
            scanner.preview_device(args.preview)
        elif args.preview_all:
            scanner.preview_all_devices()
        elif args.compare:
            scanner.compare_devices(args.compare)
        elif args.interactive:
            scanner.interactive_mode()
        else:
            # 默认显示设备列表并进入交互模式
            scanner.list_devices()
            scanner.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        scanner._stop_all_previews()


if __name__ == "__main__":
    main()