#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一推理脚本：支持TensorRT和ONNX模型
支持图像、视频和摄像头输入
优化了预处理和后处理性能
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Union
import numpy as np
import cv2

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelInference:
    """统一的模型推理接口"""
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640)):
        """
        初始化推理引擎
        
        Args:
            model_path: 模型文件路径 (.onnx 或 .engine)
            input_size: 输入图像尺寸 (width, height)
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.engine = None
        self.session = None
        self.model_type = None
        
        # 检查模型文件是否存在
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 根据文件扩展名确定模型类型
        if self.model_path.suffix.lower() == '.onnx':
            self.model_type = 'onnx'
            self._init_onnx_model()
        elif self.model_path.suffix.lower() == '.engine':
            self.model_type = 'tensorrt'
            self._init_tensorrt_model()
        else:
            raise ValueError(f"不支持的模型格式: {self.model_path.suffix}")
        
        logger.info(f"成功加载 {self.model_type.upper()} 模型: {model_path}")
        logger.info(f"输入尺寸: {input_size}")
    
    def _init_onnx_model(self):
        """初始化ONNX模型"""
        try:
            import onnxruntime as ort
            
            # 设置执行provider
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            
            # 获取输入输出信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # 打印模型信息
            input_shape = self.session.get_inputs()[0].shape
            output_shapes = [output.shape for output in self.session.get_outputs()]
            
            logger.info(f"ONNX输入形状: {input_shape}")
            logger.info(f"ONNX输出形状: {output_shapes}")
            logger.info(f"使用执行Provider: {self.session.get_providers()[0]}")
            
        except ImportError:
            raise ImportError("ONNX Runtime未安装，请安装: pip install onnxruntime-gpu")
        except Exception as e:
            raise RuntimeError(f"ONNX模型加载失败: {str(e)}")
    
    def _init_tensorrt_model(self):
        """初始化TensorRT模型"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # 创建TensorRT运行时
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            
            # 加载引擎
            with open(self.model_path, 'rb') as f:
                engine_data = f.read()
                self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                raise RuntimeError("无法反序列化TensorRT引擎")
            
            # 创建执行上下文
            self.context = self.engine.create_execution_context()
            
            # 获取输入输出信息
            self.input_shape = (1, 3, self.input_size[1], self.input_size[0])  # NCHW
            input_size_bytes = np.prod(self.input_shape) * np.dtype(np.float32).itemsize
            
            # 分配GPU内存
            self.cuda_input = cuda.mem_alloc(input_size_bytes)
            
            # 获取输出信息
            output_shapes = []
            self.cuda_outputs = []
            self.output_shapes = []
            
            for i in range(self.engine.num_bindings):
                if not self.engine.binding_is_input(i):
                    shape = self.engine.get_binding_shape(i)
                    size = trt.volume(shape) * np.dtype(np.float32).itemsize
                    self.cuda_outputs.append(cuda.mem_alloc(size))
                    self.output_shapes.append(shape)
                    output_shapes.append(shape)
            
            logger.info(f"TensorRT输入形状: {self.input_shape}")
            logger.info(f"TensorRT输出形状: {output_shapes}")
            
        except ImportError:
            raise ImportError("TensorRT或PyCUDA未安装")
        except Exception as e:
            raise RuntimeError(f"TensorRT模型加载失败: {str(e)}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            预处理后的图像数据
        """
        # 调整图像大小
        resized = cv2.resize(image, self.input_size)
        
        # BGR转RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化到[0,1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # 转换为NCHW格式
        input_data = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        input_data = np.expand_dims(input_data, 0)        # CHW -> NCHW
        
        return input_data
    
    def inference(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        模型推理
        
        Args:
            input_data: 预处理后的输入数据
            
        Returns:
            推理结果列表
        """
        if self.model_type == 'onnx':
            return self._onnx_inference(input_data)
        elif self.model_type == 'tensorrt':
            return self._tensorrt_inference(input_data)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _onnx_inference(self, input_data: np.ndarray) -> List[np.ndarray]:
        """ONNX推理"""
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        return outputs
    
    def _tensorrt_inference(self, input_data: np.ndarray) -> List[np.ndarray]:
        """TensorRT推理"""
        import pycuda.driver as cuda
        
        # 将输入数据复制到GPU
        cuda.memcpy_htod(self.cuda_input, input_data.astype(np.float32))
        
        # 创建绑定列表
        bindings = [int(self.cuda_input)]
        bindings.extend([int(cuda_out) for cuda_out in self.cuda_outputs])
        
        # 执行推理
        self.context.execute_v2(bindings)
        
        # 从GPU复制输出数据
        outputs = []
        for i, (cuda_out, shape) in enumerate(zip(self.cuda_outputs, self.output_shapes)):
            output = np.empty(shape, dtype=np.float32)
            cuda.memcpy_dtoh(output, cuda_out)
            outputs.append(output)
        
        return outputs
    
    def postprocess(self, outputs: List[np.ndarray], original_shape: Tuple[int, int], 
                   conf_threshold: float = 0.5, iou_threshold: float = 0.45) -> List[dict]:
        """
        后处理：解析模型输出，应用NMS
        
        Args:
            outputs: 模型原始输出
            original_shape: 原始图像尺寸 (height, width)
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IoU阈值
            
        Returns:
            检测结果列表，每个元素包含 {'bbox': [x1,y1,x2,y2], 'conf': float, 'class': int}
        """
        # YOLO输出格式: [batch, num_detections, 85] (4+1+80)
        # 前4个是边界框坐标(cx,cy,w,h)，第5个是置信度，后80个是类别概率
        
        predictions = outputs[0][0]  # 移除batch维度
        
        # 过滤低置信度检测
        confidences = predictions[:, 4]
        mask = confidences > conf_threshold
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            return []
        
        # 解析边界框 (cx,cy,w,h) -> (x1,y1,x2,y2)
        boxes = predictions[:, :4]
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]      # x2 = x1 + w
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]      # y2 = y1 + h
        
        # 缩放到原始图像尺寸
        scale_x = original_shape[1] / self.input_size[0]
        scale_y = original_shape[0] / self.input_size[1]
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        # 获取类别和置信度
        class_scores = predictions[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_confidences = np.max(class_scores, axis=1)
        final_confidences = predictions[:, 4] * class_confidences
        
        # 应用NMS
        indices = self.nms(boxes, final_confidences, iou_threshold)
        
        # 构建最终结果
        results = []
        for i in indices:
            results.append({
                'bbox': boxes[i].astype(int).tolist(),
                'conf': float(final_confidences[i]),
                'class': int(class_ids[i])
            })
        
        return results
    
    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        非极大值抑制
        
        Args:
            boxes: 边界框数组 [[x1,y1,x2,y2], ...]
            scores: 置信度数组
            iou_threshold: IoU阈值
            
        Returns:
            保留的索引列表
        """
        if len(boxes) == 0:
            return []
        
        # 计算面积
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 按置信度排序
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # 计算当前框与其他框的IoU
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union
            
            # 保留IoU小于阈值的框
            mask = iou <= iou_threshold
            order = order[1:][mask]
        
        return keep
    
    def predict(self, image: np.ndarray, conf_threshold: float = 0.5, 
               iou_threshold: float = 0.45) -> List[dict]:
        """
        完整的预测流程
        
        Args:
            image: 输入图像 (BGR格式)
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IoU阈值
            
        Returns:
            检测结果列表
        """
        # 预处理
        input_data = self.preprocess(image)
        
        # 推理
        outputs = self.inference(input_data)
        
        # 后处理
        results = self.postprocess(outputs, image.shape[:2], conf_threshold, iou_threshold)
        
        return results
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'cuda_input') and self.cuda_input:
            self.cuda_input.free()
        if hasattr(self, 'cuda_outputs') and self.cuda_outputs:
            for cuda_out in self.cuda_outputs:
                cuda_out.free()

class YOLODetector:
    """YOLO检测器，支持多种输入源"""
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640),
                 conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        初始化检测器
        
        Args:
            model_path: 模型文件路径
            input_size: 输入图像尺寸
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IoU阈值
        """
        self.model = ModelInference(model_path, input_size)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # COCO类别名称
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # 性能统计
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0
    
    def draw_results(self, image: np.ndarray, results: List[dict]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            results: 检测结果
            
        Returns:
            绘制了检测框的图像
        """
        annotated = image.copy()
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            conf = result['conf']
            class_id = result['class']
            
            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"Class{class_id}"
            
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated
    
    def update_fps(self):
        """更新FPS统计"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            self.avg_fps = 30 / elapsed_time
            self.fps_start_time = current_time
    
    def detect_image(self, image_path: str, output_path: Optional[str] = None):
        """
        检测单张图像
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径（可选）
        """
        logger.info(f"处理图像: {image_path}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return
        
        # 检测
        start_time = time.time()
        results = self.model.predict(image, self.conf_threshold, self.iou_threshold)
        inference_time = time.time() - start_time
        
        # 绘制结果
        annotated = self.draw_results(image, results)
        
        # 添加信息
        info_text = [
            f"检测数量: {len(results)}",
            f"推理时间: {inference_time*1000:.1f}ms",
            f"模型类型: {self.model.model_type.upper()}",
            f"图像尺寸: {image.shape[1]}x{image.shape[0]}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(annotated, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, annotated)
            logger.info(f"结果已保存到: {output_path}")
        
        # 显示结果
        cv2.imshow('Detection Result', annotated)
        logger.info("按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 打印检测结果
        logger.info(f"检测到 {len(results)} 个目标:")
        for i, result in enumerate(results):
            class_name = self.class_names[result['class']] if result['class'] < len(self.class_names) else f"Class{result['class']}"
            logger.info(f"  {i+1}. {class_name}: {result['conf']:.2f} at {result['bbox']}")
    
    def detect_video(self, video_path: str, output_path: Optional[str] = None):
        """
        检测视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
        """
        logger.info(f"处理视频: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")
        
        # 设置视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 检测
                start_time = time.time()
                results = self.model.predict(frame, self.conf_threshold, self.iou_threshold)
                inference_time = time.time() - start_time
                
                # 绘制结果
                annotated = self.draw_results(frame, results)
                
                # 更新FPS
                self.update_fps()
                
                # 添加信息
                info_text = [
                    f"帧: {frame_count}/{total_frames}",
                    f"检测: {len(results)}",
                    f"FPS: {self.avg_fps:.1f}",
                    f"推理: {inference_time*1000:.1f}ms"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(annotated, text, (10, 30 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 写入视频
                if writer:
                    writer.write(annotated)
                
                # 显示结果
                cv2.imshow('Video Detection', annotated)
                
                # 控制播放速度和退出
                key = cv2.waitKey(int(1000/fps)) & 0xFF
                if key == ord('q'):
                    break
                
        except KeyboardInterrupt:
            logger.info("检测被用户中断")
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            if output_path:
                logger.info(f"视频结果已保存到: {output_path}")
    
    def detect_camera(self, camera_id: int = 0):
        """
        摄像头实时检测
        
        Args:
            camera_id: 摄像头ID
        """
        logger.info(f"启动摄像头检测 (ID: {camera_id})")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"无法打开摄像头 {camera_id}")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("摄像头检测已启动，按 'q' 退出")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("无法读取摄像头帧")
                    break
                
                # 检测
                start_time = time.time()
                results = self.model.predict(frame, self.conf_threshold, self.iou_threshold)
                inference_time = time.time() - start_time
                
                # 绘制结果
                annotated = self.draw_results(frame, results)
                
                # 更新FPS
                self.update_fps()
                
                # 添加信息
                info_text = [
                    f"检测: {len(results)}",
                    f"FPS: {self.avg_fps:.1f}",
                    f"推理: {inference_time*1000:.1f}ms",
                    f"模型: {self.model.model_type.upper()}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(annotated, text, (10, 30 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 显示结果
                cv2.imshow('Camera Detection', annotated)
                
                # 检查退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("检测被用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='统一YOLO推理工具 - 支持TensorRT和ONNX模型')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径 (.onnx 或 .engine)')
    parser.add_argument('--source', type=str, default='0',
                       help='输入源: 图像路径、视频路径、摄像头ID(如0)或webcam')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径 (可选，仅对图像和视频有效)')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640],
                       help='模型输入尺寸 [width height] (默认: 640 640)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                       help='NMS的IoU阈值 (默认: 0.45)')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能基准测试')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("统一YOLO推理工具")
    logger.info("="*60)
    
    try:
        # 创建检测器
        detector = YOLODetector(
            model_path=args.model,
            input_size=tuple(args.input_size),
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold
        )
        
        # 性能基准测试
        if args.benchmark:
            logger.info("运行性能基准测试...")
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 预热
            for _ in range(10):
                _ = detector.model.predict(test_image)
            
            # 测试
            times = []
            for i in range(100):
                start_time = time.time()
                _ = detector.model.predict(test_image)
                times.append((time.time() - start_time) * 1000)
            
            avg_time = np.mean(times)
            fps = 1000.0 / avg_time
            
            logger.info(f"基准测试结果:")
            logger.info(f"  平均推理时间: {avg_time:.2f} ms")
            logger.info(f"  平均FPS: {fps:.1f}")
            return
        
        # 判断输入源类型
        source = args.source
        
        if source.isdigit() or source.lower() == 'webcam':
            # 摄像头输入
            camera_id = int(source) if source.isdigit() else 0
            detector.detect_camera(camera_id)
        elif source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
            # 视频文件
            detector.detect_video(source, args.output)
        else:
            # 图像文件
            detector.detect_image(source, args.output)
            
    except Exception as e:
        logger.error(f"运行失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()