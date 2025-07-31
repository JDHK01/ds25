#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson YOLO目标检测测试代码
支持摄像头实时检测和视频文件检测
"""

import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO
import torch

class JetsonYOLODetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5, device='cuda'):
        """
        初始化YOLO检测器
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.conf_threshold = conf_threshold
        self.device = device
        
        # 检查CUDA可用性
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA不可用，切换到CPU")
            self.device = 'cpu'
        # self.device = 'cpu'
        print(f"使用设备: {self.device}")
        print(f"加载模型: {model_path}")
        
        # 加载YOLO模型
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # 类别名称
        self.class_names = self.model.names
        
        # 性能统计
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0
        
    def preprocess_frame(self, frame, input_size=(640, 640)):
        """
        预处理帧
        """
        # 调整图像大小以适应Jetson性能
        height, width = frame.shape[:2]
        if width > 1280:  # 限制输入尺寸以提高性能
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame
    
    def detect_objects(self, frame):
        """
        检测目标
        """
        # 预处理
        processed_frame = self.preprocess_frame(frame)
        
        # YOLO推理
        results = self.model(processed_frame, conf=self.conf_threshold, device=self.device)
        
        return results, processed_frame
    
    def draw_results(self, frame, results):
        """
        绘制检测结果
        """
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # 绘制边界框
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 绘制标签
                    label = f"{self.class_names[cls]}: {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame
    
    def update_fps(self):
        """
        更新FPS统计
        """
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # 每30帧更新一次FPS
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            self.avg_fps = 30 / elapsed_time
            self.fps_start_time = current_time
    
    def run_camera_detection(self, camera_id=6):
        """
        运行摄像头实时检测
        """
        print(f"启动摄像头检测 (Camera ID: {camera_id})")
        
        # 初始化摄像头
        cap = cv2.VideoCapture(6)
        
        # 设置摄像头参数（针对Jetson优化）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 设置编码格式
        
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            print("尝试以下解决方案:")
            print("1. 检查摄像头是否连接正确")
            print("2. 检查摄像头权限: sudo chmod 666 /dev/video*")
            print("3. 使用 v4l2-ctl --list-devices 查看可用设备")
            return
        
        # 获取摄像头实际参数
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"摄像头参数: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        print("按 'q' 退出")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                start_time = time.time()
                
                # 检测目标
                results, processed_frame = self.detect_objects(frame)
                
                # 绘制结果
                annotated_frame = self.draw_results(processed_frame, results)
                
                # 计算推理时间
                inference_time = time.time() - start_time
                
                # 更新FPS
                self.update_fps()
                
                # 添加性能信息
                info_text = [
                    f"FPS: {self.avg_fps:.1f}",
                    f"推理时间: {inference_time*1000:.1f}ms",
                    f"设备: {self.device}",
                    f"分辨率: {annotated_frame.shape[1]}x{annotated_frame.shape[0]}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(annotated_frame, text, (10, 30 + i*25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 显示结果
                cv2.imshow('Jetson YOLO Detection', annotated_frame)
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n检测被用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def test_camera(self, camera_id=0):
        """
        测试摄像头是否可用
        """
        print(f"测试摄像头 {camera_id}...")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"摄像头 {camera_id} 无法打开")
            return False
        
        # 读取一帧测试
        ret, frame = cap.read()
        if ret:
            print(f"摄像头 {camera_id} 工作正常")
            print(f"分辨率: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
            return True
        else:
            print(f"摄像头 {camera_id} 无法读取帧")
            cap.release()
            return False
    
    def run_video_detection(self, video_path):
        """
        运行视频文件检测
        """
        print(f"处理视频文件: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("无法打开视频文件")
            return
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"视频总帧数: {total_frames}")
        print(f"视频FPS: {fps}")
        print("按 'q' 退出, 空格键暂停/继续")
        
        frame_count = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("视频播放完成")
                        break
                    frame_count += 1
                
                start_time = time.time()
                
                # 检测目标
                results, processed_frame = self.detect_objects(frame)
                
                # 绘制结果
                annotated_frame = self.draw_results(processed_frame, results)
                
                # 计算推理时间
                inference_time = time.time() - start_time
                
                # 添加信息
                info_text = [
                    f"帧: {frame_count}/{total_frames}",
                    f"推理时间: {inference_time*1000:.1f}ms",
                    f"设备: {self.device}",
                    "空格键: 暂停/继续, Q: 退出"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(annotated_frame, text, (10, 30 + i*25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 显示结果
                cv2.imshow('Jetson YOLO Video Detection', annotated_frame)
                
                # 控制播放速度
                key = cv2.waitKey(int(1000/fps)) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    
        except KeyboardInterrupt:
            print("\n检测被用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Jetson YOLO目标检测')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='YOLO模型路径 (默认: yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.5, 
                       help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--source', type=str, default='0', 
                       help='输入源: camera, 摄像头ID(如6), 或视频文件路径')
    parser.add_argument('--test-camera', action='store_true',
                       help='仅测试摄像头是否可用，不运行检测')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Jetson YOLO目标检测系统")
    print("=" * 50)
    
    # 创建检测器
    detector = JetsonYOLODetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device
    )
    
    # 如果只是测试摄像头
    if args.test_camera:
        if args.source == 'camera' or args.source.isdigit():
            camera_id = 0 if args.source == 'camera' else int(args.source)
            detector.test_camera(camera_id)
        else:
            print("--test-camera 选项只能用于摄像头测试")
        return
    
    # 根据输入源类型运行检测
    if args.source == 'camera' or args.source.isdigit():
        camera_id = 0 if args.source == 'camera' else int(args.source)
        detector.run_camera_detection(camera_id)
    else:
        detector.run_video_detection(args.source)

if __name__ == "__main__":
    main()
