#!/usr/bin/env python3
"""
视觉裁切测试脚本
测试drone_ctrl.py中的视野裁切功能
"""

import cv2
import numpy as np
import sys
import os

# 添加路径以导入检测器（如果可用）
sys.path.append("/Users/yqz/by/temp/vision/yolo")

def test_crop_region():
    """测试裁切区域功能"""
    print("=== 视觉裁切测试 ===")
    
    # 尝试打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("警告: 无法打开摄像头，使用模拟图像测试")
        return test_with_synthetic_image()
    
    # 设置相机参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("按键说明:")
    print("- 空格键: 暂停/继续")
    print("- 's': 保存当前帧")
    print("- 'q': 退出")
    print("- 'r': 切换裁切区域显示")
    
    paused = False
    show_crop_region = True
    frame_count = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break
                frame_count += 1
            
            # 创建显示帧的副本
            display_frame = frame.copy()
            
            # 定义裁切区域
            crop_x1, crop_y1 = 180, 50
            crop_x2, crop_y2 = 540, 410
            
            # 裁切区域
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # 在原始帧上绘制裁切区域框
            if show_crop_region:
                # 绘制裁切区域边框（绿色）
                cv2.rectangle(display_frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)
                
                # 添加文本标签
                cv2.putText(display_frame, f"Crop Region: ({crop_x1},{crop_y1}) to ({crop_x2},{crop_y2})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Crop Size: {crop_x2-crop_x1}x{crop_y2-crop_y1}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 在裁切区域中心绘制十字线
                center_x = (crop_x1 + crop_x2) // 2
                center_y = (crop_y1 + crop_y2) // 2
                cv2.line(display_frame, (center_x-20, center_y), (center_x+20, center_y), (0, 255, 0), 1)
                cv2.line(display_frame, (center_x, center_y-20), (center_x, center_y+20), (0, 255, 0), 1)
            
            # 显示状态信息
            status_text = "PAUSED" if paused else f"Frame: {frame_count}"
            cv2.putText(display_frame, status_text, (10, display_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示窗口
            cv2.imshow('Original Frame with Crop Region', display_frame)
            cv2.imshow('Cropped Region', cropped_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键暂停/继续
                paused = not paused
                print(f"{'暂停' if paused else '继续'}")
            elif key == ord('s'):  # 保存帧
                cv2.imwrite(f'test_frame_{frame_count:04d}.jpg', display_frame)
                cv2.imwrite(f'test_cropped_{frame_count:04d}.jpg', cropped_frame)
                print(f"保存帧 {frame_count}")
            elif key == ord('r'):  # 切换裁切区域显示
                show_crop_region = not show_crop_region
                print(f"裁切区域显示: {'开' if show_crop_region else '关'}")
                
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("测试完成")

def test_with_synthetic_image():
    """使用合成图像测试裁切功能"""
    print("使用合成图像测试...")
    
    # 创建640x480的测试图像
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 绘制网格
    for i in range(0, 640, 40):
        cv2.line(frame, (i, 0), (i, 480), (50, 50, 50), 1)
    for i in range(0, 480, 40):
        cv2.line(frame, (0, i), (640, i), (50, 50, 50), 1)
    
    # 绘制一些彩色形状作为测试对象
    cv2.circle(frame, (100, 100), 30, (0, 0, 255), -1)  # 红色圆
    cv2.rectangle(frame, (200, 150), (280, 200), (0, 255, 0), -1)  # 绿色矩形
    cv2.circle(frame, (350, 250), 25, (255, 0, 0), -1)  # 蓝色圆
    cv2.rectangle(frame, (450, 300), (550, 380), (255, 255, 0), -1)  # 青色矩形
    
    # 定义裁切区域
    crop_x1, crop_y1 = 180, 50
    crop_x2, crop_y2 = 540, 410
    
    # 裁切
    cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # 绘制裁切区域框
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)
    cv2.putText(display_frame, f"Crop: ({crop_x1},{crop_y1}) to ({crop_x2},{crop_y2})", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow('Synthetic Test - Original with Crop Region', display_frame)
    cv2.imshow('Synthetic Test - Cropped Region', cropped)
    
    print("按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return True

def test_detection_integration():
    """测试与检测器的集成"""
    print("\n=== 检测器集成测试 ===")
    
    try:
        # 尝试导入检测器
        from detect import YOLOv8AnimalDetector
        
        # 检查模型文件是否存在
        model_path = "/Users/yqz/by/temp/vision/yolo/best9999.onnx"
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在: {model_path}")
            return False
        
        detector = YOLOv8AnimalDetector(model_path)
        print("检测器初始化成功")
        
        # 测试检测功能
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头进行检测测试")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("检测测试开始... 按 'q' 退出")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 裁切区域
            cropped_frame = frame[50:410, 180:540]
            
            # 检测
            result = detector.detect_animals(cropped_frame, show_result=False)
            
            # 绘制裁切区域
            cv2.rectangle(frame, (180, 50), (540, 410), (0, 255, 0), 2)
            
            # 如果有检测结果，转换坐标并绘制
            if result:
                for animal_type, boxes in result.items():
                    for box in boxes:
                        if len(box) >= 4:
                            x1, y1, x2, y2 = box[:4]
                            # 转换回原始图像坐标
                            orig_x1 = x1 + 180
                            orig_y1 = y1 + 50
                            orig_x2 = x2 + 180
                            orig_y2 = y2 + 50
                            
                            # 绘制检测框
                            cv2.rectangle(frame, (int(orig_x1), int(orig_y1)), 
                                        (int(orig_x2), int(orig_y2)), (255, 0, 0), 2)
                            cv2.putText(frame, f"{animal_type}", (int(orig_x1), int(orig_y1-10)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow('Detection Test with Crop', frame)
            cv2.imshow('Cropped Detection Region', cropped_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return True
        
    except ImportError:
        print("无法导入检测器模块")
        return False
    except Exception as e:
        print(f"检测器测试失败: {e}")
        return False

if __name__ == "__main__":
    print("视觉裁切测试工具")
    print("==================")
    
    # 基础裁切测试
    test_crop_region()
    
    # 检测器集成测试
    test_detection_integration()
    
    print("\n所有测试完成!")