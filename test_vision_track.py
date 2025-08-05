#!/usr/bin/env python3
"""
测试视觉追踪的代码
专门用于测试drone_ctrl中的视觉追踪逻辑

包含以下测试场景:
1. 基础追踪逻辑测试 (无真实无人机)
2. 模拟无人机的追踪测试
3. PID控制器测试
4. 状态转换测试
5. 完整追踪流程测试
"""

import asyncio
import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from unittest.mock import Mock, AsyncMock

# 添加路径
sys.path.append(str(Path(__file__).parent / "mycontrol"))
sys.path.append(str(Path(__file__).parent / "vision/yolo"))
sys.path.append(str(Path(__file__).parent / "util"))

try:
    from drone_ctrl import Drone_Controller, CameraConfig, DroneCommand, TaskState, PIDController
    from detect import YOLOv8AnimalDetector
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖模块都在正确路径下")
    sys.exit(1)

class MockDetector:
    """模拟检测器，用于测试"""
    
    def __init__(self, simulate_detections=True):
        self.simulate_detections = simulate_detections
        self.detection_count = 0
        
    def detect_animals(self, frame, show_result=False):
        """模拟动物检测"""
        if not self.simulate_detections:
            return {}
        
        self.detection_count += 1
        
        # 模拟检测结果 - 返回字典格式，包含动物类型和边界框列表
        if self.detection_count % 3 == 0:  # 每3帧返回一次检测结果
            # 模拟在画面中心附近检测到一只老虎
            return {
                'tiger': [
                    [280, 200, 360, 280, 0.85]  # [x1, y1, x2, y2, confidence]
                ]
            }
        elif self.detection_count % 5 == 0:  # 每5帧检测到大象
            return {
                'elephant': [
                    [250, 180, 390, 300, 0.92]
                ]
            }
        else:
            return {}  # 未检测到动物

class MockDrone:
    """模拟无人机类，用于测试"""
    
    def __init__(self):
        self.position = {'north': 0.0, 'east': 0.0, 'down': -1.2}
        self.velocity_commands = []
        self.offboard = Mock()
        self.offboard.set_velocity_body = AsyncMock(side_effect=self._set_velocity_body)
        
    async def _set_velocity_body(self, velocity_body_yawspeed):
        """记录速度命令"""
        command = {
            'timestamp': time.time(),
            'forward': velocity_body_yawspeed.forward_m_s,
            'right': velocity_body_yawspeed.right_m_s,
            'down': velocity_body_yawspeed.down_m_s,
            'yaw_speed': velocity_body_yawspeed.yawspeed_deg_s
        }
        self.velocity_commands.append(command)
        
        # 模拟位置更新
        dt = 0.02  # 50Hz控制频率
        self.position['north'] += velocity_body_yawspeed.forward_m_s * dt
        self.position['east'] += velocity_body_yawspeed.right_m_s * dt
        self.position['down'] += velocity_body_yawspeed.down_m_s * dt
        
    def get_velocity_history(self):
        """获取速度命令历史"""
        return self.velocity_commands.copy()
    
    def clear_velocity_history(self):
        """清空速度命令历史"""
        self.velocity_commands.clear()

def create_test_frame_with_object(width=640, height=480, object_center=(320, 240), object_size=(80, 80)):
    """创建包含模拟目标的测试图像"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 添加背景噪声
    frame = cv2.randu(frame, 0, 50)
    
    # 在指定位置绘制目标对象 (模拟动物)
    x, y = object_center
    w, h = object_size
    
    # 绘制矩形目标
    cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), -1)
    
    # 添加一些特征点
    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    
    return frame

class TestSuite:
    """视觉追踪测试套件"""
    
    def __init__(self):
        self.test_results = {}
        
    async def test_basic_tracking_logic(self):
        """测试基础追踪逻辑"""
        print("\n=== 测试1: 基础追踪逻辑 ===")
        
        try:
            # 创建控制器
            camera_config = CameraConfig(
                width=640, 
                height=480, 
                show_window=False,
                device_id=0  # 使用默认设备ID
            )
            
            controller = Drone_Controller(
                path_label=["A9B1"], 
                camera_config=camera_config
            )
            
            # 创建模拟检测器
            detector = MockDetector(simulate_detections=True)
            
            # 测试几帧处理
            test_frames = 5
            commands_generated = 0
            
            print(f"处理 {test_frames} 帧测试...")
            
            for i in range(test_frames):
                # 创建测试帧
                if i % 2 == 0:
                    # 偶数帧有目标
                    frame = create_test_frame_with_object(
                        object_center=(320 + i*10, 240 + i*5)  # 目标稍微移动
                    )
                else:
                    # 奇数帧无目标
                    frame = create_test_frame_with_object(
                        object_center=(-100, -100)  # 目标在画面外
                    )
                
                # 模拟相机读取 - 直接设置帧
                if controller.camera is None:
                    controller.camera = Mock()
                    controller.camera.read = Mock(return_value=(True, frame))
                else:
                    controller.camera.read.return_value = (True, frame)
                
                frame_result, command = controller.process_tracking_frame(detector)
                
                if command is not None and (command.velocity_forward != 0 or command.velocity_right != 0):
                    commands_generated += 1
                    print(f"  帧 {i}: 生成追踪命令 - 前进:{command.velocity_forward:.3f}, 右移:{command.velocity_right:.3f}")
                else:
                    print(f"  帧 {i}: 悬停或无目标")
                
                # 获取追踪状态
                tracking_info = controller.get_tracking_info()
                print(f"    状态: {tracking_info['state']}, 跟踪中: {tracking_info['tracking']}")
                
                await asyncio.sleep(0.02)  # 模拟50Hz
            
            # 清理资源
            controller.cleanup_camera()
            
            success = commands_generated > 0
            self.test_results['basic_tracking'] = success
            print(f"✅ 基础追踪逻辑测试: {'通过' if success else '失败'}")
            print(f"   生成了 {commands_generated} 个有效追踪命令")
            
        except Exception as e:
            print(f"❌ 基础追踪逻辑测试失败: {e}")
            self.test_results['basic_tracking'] = False

    async def test_pid_controller(self):
        """测试PID控制器"""
        print("\n=== 测试2: PID控制器 ===")
        
        try:
            # 创建PID控制器
            pid = PIDController(kp=0.3, ki=0.1, kd=0.05, output_limit=1.0)
            
            # 测试不同误差值
            test_errors = [0.5, 0.3, 0.1, 0.0, -0.1, -0.3, -0.5]
            outputs = []
            
            print("PID响应测试:")
            for error in test_errors:
                output = pid.compute(error)
                outputs.append(output)
                print(f"  误差: {error:+6.2f} -> 输出: {output:+6.3f}")
                time.sleep(0.02)  # 模拟时间间隔
            
            # 检查输出是否合理
            success = all(abs(output) <= 1.0 for output in outputs)  # 输出应在限制范围内
            
            # 重置测试
            pid.reset()
            output_after_reset = pid.compute(0.5)
            print(f"重置后首次输出: {output_after_reset:.3f}")
            
            self.test_results['pid_controller'] = success
            print(f"✅ PID控制器测试: {'通过' if success else '失败'}")
            
        except Exception as e:
            print(f"❌ PID控制器测试失败: {e}")
            self.test_results['pid_controller'] = False

    async def test_state_transitions(self):
        """测试状态转换逻辑"""
        print("\n=== 测试3: 状态转换 ===")
        
        try:
            camera_config = CameraConfig(show_window=False)
            controller = Drone_Controller(
                path_label=["A9B1"], 
                camera_config=camera_config,
                navigation_config={
                    'completion_tolerance': 50,  # 更大的容差便于测试
                    'alignment_duration': 1.0     # 缩短对准时间
                }
            )
            
            # 重置状态
            controller.reset_tracking_task()
            print(f"初始状态: {controller.get_task_state().value}")
            
            # 模拟检测器
            detector = MockDetector(simulate_detections=True)
            
            # 模拟相机
            controller.camera = Mock()
            
            # 测试状态转换: TRACKING -> APPROACHING -> COMPLETED
            states_observed = []
            
            for i in range(100):  # 多帧测试
                # 创建目标逐渐接近中心的帧
                if i < 50:
                    # 前50帧：目标在画面边缘，应该是TRACKING状态
                    target_x = 320 + 100 - i * 2  # 从边缘向中心移动
                    target_y = 240
                else:
                    # 后50帧：目标在中心附近，应该转换到APPROACHING然后COMPLETED
                    target_x = 320 + np.random.randint(-20, 21)  # 在中心附近随机移动
                    target_y = 240 + np.random.randint(-20, 21)
                
                frame = create_test_frame_with_object(object_center=(target_x, target_y))
                controller.camera.read.return_value = (True, frame)
                
                frame_result, command = controller.process_tracking_frame(detector)
                current_state = controller.get_task_state()
                
                if len(states_observed) == 0 or states_observed[-1] != current_state:
                    states_observed.append(current_state)
                    print(f"  帧 {i}: 状态转换到 {current_state.value}")
                
                # 如果已经完成，跳出循环
                if controller.is_task_completed():
                    print(f"  任务在第 {i} 帧完成")
                    break
                
                await asyncio.sleep(0.01)  # 加快测试速度
            
            # 检查是否观察到了预期的状态转换
            state_values = [state.value for state in states_observed]
            expected_transitions = ['tracking', 'approaching', 'completed']
            
            success = all(state in state_values for state in expected_transitions)
            
            controller.cleanup_camera()
            
            self.test_results['state_transitions'] = success
            print(f"✅ 状态转换测试: {'通过' if success else '失败'}")
            print(f"   观察到的状态序列: {state_values}")
            
        except Exception as e:
            print(f"❌ 状态转换测试失败: {e}")
            self.test_results['state_transitions'] = False

    async def test_tracking_with_mock_drone(self):
        """测试与模拟无人机的集成追踪"""
        print("\n=== 测试4: 模拟无人机追踪 ===")
        
        try:
            # 创建模拟无人机
            mock_drone = MockDrone()
            
            # 创建控制器
            camera_config = CameraConfig(show_window=False)
            controller = Drone_Controller(
                path_label=["A9B1"], 
                camera_config=camera_config,
                navigation_config={
                    'completion_tolerance': 30,
                    'alignment_duration': 0.5  # 缩短测试时间
                }
            )
            
            # 创建检测器
            detector = MockDetector(simulate_detections=True)
            
            print("开始模拟追踪任务...")
            tracking_result = await controller.visual_tracking_mode(
                mock_drone, detector, duration=2.0  # 2秒测试
            )
            
            # 分析无人机的运动历史
            velocity_history = mock_drone.get_velocity_history()
            
            print(f"追踪结果: {'成功' if tracking_result else '超时'}")
            print(f"生成了 {len(velocity_history)} 个速度命令")
            
            if velocity_history:
                # 分析运动特征
                non_zero_commands = [cmd for cmd in velocity_history 
                                   if abs(cmd['forward']) > 0.01 or abs(cmd['right']) > 0.01]
                print(f"其中 {len(non_zero_commands)} 个为非零运动命令")
                
                if non_zero_commands:
                    avg_forward = np.mean([cmd['forward'] for cmd in non_zero_commands])
                    avg_right = np.mean([cmd['right'] for cmd in non_zero_commands])
                    print(f"平均速度: 前进={avg_forward:.3f}, 右移={avg_right:.3f}")
            
            success = len(velocity_history) > 0 and len(non_zero_commands) > 0
            
            self.test_results['mock_drone_tracking'] = success
            print(f"✅ 模拟无人机追踪测试: {'通过' if success else '失败'}")
            
        except Exception as e:
            print(f"❌ 模拟无人机追踪测试失败: {e}")
            self.test_results['mock_drone_tracking'] = False

    async def test_real_detector_integration(self):
        """测试真实检测器集成 (如果模型文件存在)"""
        print("\n=== 测试5: 真实检测器集成 ===")
        
        try:
            # 检查模型文件是否存在
            model_path = "./vision/yolo/best9999.onnx"
            if not os.path.exists(model_path):
                print(f"⚠️  模型文件不存在: {model_path}")
                print("   跳过真实检测器测试")
                self.test_results['real_detector'] = None
                return
            
            # 创建真实检测器
            try:
                detector = YOLOv8AnimalDetector(model_path)
                print("✅ 真实检测器加载成功")
            except Exception as e:
                print(f"❌ 检测器加载失败: {e}")
                self.test_results['real_detector'] = False
                return
            
            # 创建控制器
            camera_config = CameraConfig(show_window=False)
            controller = Drone_Controller(
                path_label=["A9B1"], 
                camera_config=camera_config
            )
            
            # 模拟相机和测试图像
            controller.camera = Mock()
            
            # 创建包含动物的测试图像 (使用更真实的图像)
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # 在图像中心画一个更像动物的形状
            cv2.ellipse(test_frame, (320, 240), (60, 40), 0, 0, 360, (139, 69, 19), -1)  # 棕色椭圆
            
            controller.camera.read.return_value = (True, test_frame)
            
            # 测试检测
            frame_result, command = controller.process_tracking_frame(detector)
            
            print("真实检测器测试完成")
            
            success = frame_result is not None
            self.test_results['real_detector'] = success
            print(f"✅ 真实检测器集成测试: {'通过' if success else '失败'}")
            
            controller.cleanup_camera()
            
        except Exception as e:
            print(f"❌ 真实检测器集成测试失败: {e}")
            self.test_results['real_detector'] = False

    async def run_all_tests(self):
        """运行所有测试"""
        print("🚁 开始视觉追踪测试套件")
        print("=" * 50)
        
        start_time = time.time()
        
        await self.test_basic_tracking_logic()
        await self.test_pid_controller()
        await self.test_state_transitions()
        await self.test_tracking_with_mock_drone()
        await self.test_real_detector_integration()
        
        end_time = time.time()
        
        # 输出测试结果摘要
        print("\n" + "=" * 50)
        print("📊 测试结果摘要")
        print("=" * 50)
        
        passed = 0
        total = 0
        
        for test_name, result in self.test_results.items():
            if result is not None:
                total += 1
                if result:
                    passed += 1
                    status = "✅ 通过"
                else:
                    status = "❌ 失败"
            else:
                status = "⚠️  跳过"
            
            print(f"{test_name:25}: {status}")
        
        print("-" * 50)
        if total > 0:
            print(f"总计: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
        print(f"测试耗时: {end_time - start_time:.2f} 秒")
        
        return passed, total

def main():
    """主函数"""
    print("🎯 无人机视觉追踪测试工具")
    print("测试 drone_ctrl.py 中的视觉追踪逻辑")
    print()
    
    # 创建测试套件
    test_suite = TestSuite()
    
    try:
        # 运行所有测试
        passed, total = asyncio.run(test_suite.run_all_tests())
        
        # 退出代码
        if total == 0:
            exit_code = 0  # 没有测试运行
        elif passed == total:
            exit_code = 0  # 所有测试通过
        else:
            exit_code = 1  # 有测试失败
        
        print(f"\n程序退出，代码: {exit_code}")
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
        return 130
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)