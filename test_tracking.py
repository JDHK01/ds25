#!/usr/bin/env python3
"""
视觉跟踪测试脚本
测试集成到drone_ctrl.py中的跟踪功能
"""

import sys
import asyncio
import time
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent / "mycontrol"))
sys.path.append(str(Path(__file__).parent / "vision/yolo"))
sys.path.append(str(Path(__file__).parent / "lib"))

from drone_ctrl import Drone_Controller, CameraConfig, TargetMode, TaskState
from detect import YOLOv8AnimalDetector
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed


class MockDrone:
    """模拟无人机类用于测试"""
    def __init__(self):
        self.position = {'x': 0.0, 'y': 0.0, 'z': -1.0}
        self.velocity = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
    class Offboard:
        def __init__(self, drone):
            self.drone = drone
            
        async def set_velocity_body(self, velocity_body):
            # 模拟速度控制
            self.drone.velocity['x'] = velocity_body.forward_m_s
            self.drone.velocity['y'] = velocity_body.right_m_s  
            self.drone.velocity['z'] = velocity_body.down_m_s
            print(f"[模拟无人机] 设置速度: 前进={velocity_body.forward_m_s:.3f}, 右移={velocity_body.right_m_s:.3f}, 下降={velocity_body.down_m_s:.3f}")
            
    def __init__(self):
        self.position = {'x': 0.0, 'y': 0.0, 'z': -1.0}
        self.velocity = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.offboard = self.Offboard(self)


async def test_tracking_only():
    """纯跟踪功能测试（无无人机连接）"""
    print("=" * 60)
    print("视觉跟踪功能测试（无无人机模式）")
    print("=" * 60)
    
    # 相机配置
    camera_config = CameraConfig(
        width=640,
        height=480,
        fps=30,
        device_id=0,
        buffer_size=1,
        # 相机偏移配置（单位：米）
        offset_forward=0.1,   # 相机在无人机前方10cm
        offset_right=0.0,     # 相机在中心线上
        offset_down=0.05,     # 相机在无人机下方5cm
        # 窗口显示配置
        show_window=False     # 设为True可以看到实时画面
    )
    
    # 导航配置
    navigation_config = {
        'position_tolerance': 20,
        'min_target_area': 1000,
        'max_velocity': 0.5,
        'offset_compensation_gain': 0.3,
        'alignment_duration': 2.0,
        'completion_tolerance': 15
    }
    
    # PID配置
    pid_config = {
        'horizontal': {
            'kp': 0.5,
            'ki': 0.0,
            'kd': 0.1,
            'output_limit': 0.5
        }
    }
    
    # 创建控制器
    controller = Drone_Controller(
        path_label=["A9B1"],
        camera_config=camera_config,
        navigation_config=navigation_config,
        pid_config=pid_config
    )
    
    # 创建检测器
    try:
        detector = YOLOv8AnimalDetector('./vision/yolo/best9999.onnx')
        print("✓ 检测器初始化成功")
    except Exception as e:
        print(f"✗ 检测器初始化失败: {e}")
        print("使用模拟检测器")
        detector = None
    
    # 测试跟踪处理
    print("\n开始跟踪测试...")
    print("按 Ctrl+C 退出测试")
    
    test_start_time = time.time()
    test_duration = 10.0  # 测试10秒
    
    try:
        while time.time() - test_start_time < test_duration:
            if detector:
                # 真实检测器测试
                frame, command = controller.process_tracking_frame(detector)
            else:
                # 模拟检测结果测试
                frame, command = None, None
                print("[模拟] 未检测到目标")
            
            if command:
                elapsed_time = time.time() - test_start_time
                print(f"[{elapsed_time:6.2f}s] 跟踪命令: 前进={command.velocity_forward:+6.3f}, 右移={command.velocity_right:+6.3f}, 下降={command.velocity_down:+6.3f}")
            else:
                elapsed_time = time.time() - test_start_time
                print(f"[{elapsed_time:6.2f}s] 无跟踪命令")
            
            # 显示跟踪状态
            tracking_info = controller.get_tracking_info()
            if tracking_info['state'] != 'tracking':
                print(f"跟踪状态: {tracking_info['state']}")
                if 'approaching_progress' in tracking_info:
                    print(f"  靠近进度: {tracking_info['approaching_progress']*100:.0f}%")
            
            # 检查任务完成 
            if controller.is_task_completed():
                print("\n✓ 跟踪任务完成!")
                break
                
            time.sleep(0.1)  # 10Hz测试频率
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    finally:
        controller.cleanup_camera()
        
    print(f"\n跟踪测试结束，持续时间: {time.time() - test_start_time:.2f}秒")


async def test_tracking_with_mock_drone():
    """带模拟无人机的跟踪测试"""
    print("=" * 60)
    print("视觉跟踪功能测试（模拟无人机模式）")
    print("=" * 60)
    
    # 配置（同上）
    camera_config = CameraConfig(
        width=640,
        height=480,
        device_id=0,
        show_window=False
    )
    
    navigation_config = {
        'position_tolerance': 20,
        'min_target_area': 1000,
        'max_velocity': 0.5,
        'alignment_duration': 3.0,
        'completion_tolerance': 15
    }
    
    pid_config = {
        'horizontal': {
            'kp': 0.3,
            'ki': 0.0,
            'kd': 0.05,
            'output_limit': 0.3
        }
    }
    
    # 创建控制器和模拟无人机
    controller = Drone_Controller(
        path_label=["A9B1"],
        camera_config=camera_config,
        navigation_config=navigation_config,
        pid_config=pid_config
    )
    
    mock_drone = MockDrone()
    
    # 创建检测器
    try:
        detector = YOLOv8AnimalDetector('./vision/yolo/best9999.onnx')
        print("✓ 检测器初始化成功")
    except Exception as e:
        print(f"✗ 检测器初始化失败: {e}")
        return
    
    # 执行跟踪任务
    print("\n开始跟踪任务（5秒超时）...")
    tracking_result = await controller.visual_tracking_mode(mock_drone, detector, duration=5.0)
    
    if tracking_result:
        print("✓ 跟踪任务成功完成!")
    else:
        print("✗ 跟踪任务超时未完成")
    
    # 清理
    controller.cleanup_camera()


async def test_integrated_tracking():
    """集成跟踪测试 - 测试与现有路径规划的集成"""
    print("=" * 60)
    print("集成跟踪测试")
    print("=" * 60)
    
    # 创建带跟踪功能的控制器
    camera_config = CameraConfig(device_id=0, show_window=False)
    controller = Drone_Controller(
        path_label=["A9B1", "A8B1"],
        camera_config=camera_config
    )
    
    mock_drone = MockDrone()
    
    # 测试各种跟踪功能
    print("测试1: 跟踪状态管理")
    print(f"初始状态: {controller.get_task_state().value}")
    
    print("测试2: 重置跟踪任务")
    controller.reset_tracking_task()
    print(f"重置后状态: {controller.get_task_state().value}")
    
    print("测试3: 跟踪信息获取")
    info = controller.get_tracking_info()
    print(f"跟踪信息: {info}")
    
    print("测试4: 相机初始化")
    try:
        controller.init_camera()
        print("✓ 相机初始化成功")
    except Exception as e:
        print(f"✗ 相机初始化失败: {e}")
    
    print("测试5: 清理资源")
    controller.cleanup_camera()
    print("✓ 资源清理完成")
    
    print("\n✓ 集成跟踪测试完成")


def main():
    """主测试函数"""
    print("视觉跟踪功能测试套件")
    print("请选择测试模式:")
    print("1. 纯跟踪功能测试（无无人机）")
    print("2. 模拟无人机跟踪测试")
    print("3. 集成功能测试")
    print("4. 运行所有测试")
    
    try:
        choice = input("请输入选择 (1-4): ").strip()
        
        if choice == '1':
            asyncio.run(test_tracking_only())
        elif choice == '2':
            asyncio.run(test_tracking_with_mock_drone())
        elif choice == '3':
            asyncio.run(test_integrated_tracking())
        elif choice == '4':
            print("\n运行所有测试...")
            asyncio.run(test_integrated_tracking())
            asyncio.run(test_tracking_only())
            asyncio.run(test_tracking_with_mock_drone())
        else:
            print("无效选择，运行默认测试")
            asyncio.run(test_integrated_tracking())
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")


if __name__ == "__main__":
    main()