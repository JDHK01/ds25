#!/usr/bin/env python3
"""
TensorRT动物检测API
优化版本 - 包含大象尺寸优化和简单统计接口
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import base64
from io import BytesIO
from PIL import Image


class TensorRTAnimalDetectionAPI:
    def __init__(self, engine_path: str, class_names: Optional[Dict[int, str]] = None):
        """
        初始化TensorRT动物检测API
        
        Args:
            engine_path: TensorRT engine文件路径
            class_names: 类别名称字典
        """
        self.engine_path = engine_path
        self.class_names = class_names or {
            0: 'elephant', 1: 'monkey', 2: 'peacock', 3: 'wolf', 4: 'tiger'
        }
        
        # 动物尺寸信息 - 大象是其他动物的两倍
        self.animal_size_ratios = {
            'elephant': 2.0,    # 大象是基准的2倍
            'monkey': 1.0,      # 基准尺寸
            'peacock': 1.0,     # 基准尺寸
            'wolf': 1.0,        # 基准尺寸
            'tiger': 1.0        # 基准尺寸
        }
        
        # 基准面积范围（针对基准动物）
        self.base_min_area = 400
        self.base_max_area = 50000
        
        # 初始化TensorRT
        self._init_tensorrt()
        
        # 性能统计
        self.total_inferences = 0
        self.total_time = 0.0
        
    def _init_tensorrt(self):
        """初始化TensorRT组件"""
        print(f"加载TensorRT Engine: {self.engine_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 加载engine
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # 设置输入输出绑定
        self._setup_bindings()
        print("TensorRT初始化完成")
        
    def _setup_bindings(self):
        """设置输入输出绑定"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape) * self.engine.max_batch_size
            
            # 分配GPU内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(i):
                self.input_shape = shape
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.output_shape = shape
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                
    def preprocess_image(self, image: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        预处理图像
        
        Args:
            image: 输入图像 (BGR格式)
            target_size: 目标尺寸
            
        Returns:
            处理后的图像数组、缩放比例、偏移量
        """
        h, w, c = image.shape
        
        # 计算缩放比例
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建目标尺寸的画布并填充
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        
        # 计算填充位置
        start_y = (target_size - new_h) // 2
        start_x = (target_size - new_w) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        # 转换为RGB并归一化
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas = canvas.astype(np.float32) / 255.0
        
        # 转换维度顺序 HWC -> CHW
        canvas = np.transpose(canvas, (2, 0, 1))
        
        # 添加batch维度
        canvas = np.expand_dims(canvas, axis=0)
        
        return canvas, scale, (start_x, start_y)
    
    def inference(self, input_data: np.ndarray) -> np.ndarray:
        """执行推理"""
        # 复制输入数据到GPU
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 复制输出数据到CPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        # 重塑输出数据
        output_data = self.outputs[0]['host'].reshape(self.output_shape)
        return output_data
    
    def _get_area_threshold_for_animal(self, class_name: str) -> Tuple[int, int]:
        """
        根据动物类型获取面积阈值
        
        Args:
            class_name: 动物类别名称
            
        Returns:
            (最小面积, 最大面积)
        """
        size_ratio = self.animal_size_ratios.get(class_name, 1.0)
        
        # 根据尺寸比例调整面积阈值
        min_area = int(self.base_min_area * (size_ratio ** 0.8))  # 稍微缓和尺寸差异
        max_area = int(self.base_max_area * (size_ratio ** 1.2))  # 大象可以更大
        
        return min_area, max_area
    
    def postprocess(self, predictions: np.ndarray, original_shape: Tuple[int, int], 
                   scale: float, offset: Tuple[int, int], 
                   conf_threshold: float = 0.6, nms_threshold: float = 0.3) -> List[Dict]:
        """
        优化的后处理 - 包含动物尺寸优化
        """
        results = []
        
        # 解析预测结果
        if len(predictions.shape) == 3:
            predictions = predictions[0]
            
        if predictions.shape[0] == 9:
            predictions = predictions.T
            
        # 解析数据
        boxes = predictions[:, :4]
        class_probs = predictions[:, 4:9]
        
        # 应用Sigmoid激活
        class_probs = self._sigmoid(class_probs)
        
        # 获取最大类别概率和类别ID
        max_class_probs = np.max(class_probs, axis=1)
        class_ids = np.argmax(class_probs, axis=1)
        
        # 置信度过滤
        conf_mask = max_class_probs > conf_threshold
        if conf_mask.sum() == 0:
            return results
            
        boxes = boxes[conf_mask]
        max_class_probs = max_class_probs[conf_mask]
        class_ids = class_ids[conf_mask]
        
        # 转换bbox格式
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        boxes_xyxy = np.column_stack([x1, y1, x2, y2])
        
        # 边界检查
        target_size = 640
        valid_mask = (
            (boxes_xyxy[:, 0] >= 0) & (boxes_xyxy[:, 1] >= 0) &
            (boxes_xyxy[:, 2] <= target_size) & (boxes_xyxy[:, 3] <= target_size) &
            (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        )
        
        boxes_xyxy = boxes_xyxy[valid_mask]
        max_class_probs = max_class_probs[valid_mask]
        class_ids = class_ids[valid_mask]
        
        if len(boxes_xyxy) == 0:
            return results
        
        # 基于动物类型的面积过滤
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        valid_indices = []
        
        for i, (class_id, area) in enumerate(zip(class_ids, areas)):
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            min_area, max_area = self._get_area_threshold_for_animal(class_name)
            
            if min_area <= area <= max_area:
                valid_indices.append(i)
        
        if not valid_indices:
            return results
            
        valid_indices = np.array(valid_indices)
        boxes_xyxy = boxes_xyxy[valid_indices]
        max_class_probs = max_class_probs[valid_indices]
        class_ids = class_ids[valid_indices]
        
        # 宽高比过滤
        widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        aspect_ratios = widths / heights
        
        # 根据动物调整宽高比范围
        aspect_mask = np.ones(len(boxes_xyxy), dtype=bool)
        for i, class_id in enumerate(class_ids):
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            if class_name == 'elephant':
                # 大象可能更宽或更高
                aspect_mask[i] = 0.1 <= aspect_ratios[i] <= 8.0
            else:
                # 其他动物的标准范围
                aspect_mask[i] = 0.2 <= aspect_ratios[i] <= 5.0
        
        boxes_xyxy = boxes_xyxy[aspect_mask]
        max_class_probs = max_class_probs[aspect_mask]
        class_ids = class_ids[aspect_mask]
        
        if len(boxes_xyxy) == 0:
            return results
        
        # 应用NMS
        indices = self._nms(boxes_xyxy, max_class_probs, nms_threshold)
        
        # 转换坐标到原始图像
        start_x, start_y = offset
        original_h, original_w = original_shape
        
        for i in indices:
            x1, y1, x2, y2 = boxes_xyxy[i]
            
            # 还原到原始图像坐标
            x1 = max(0, int((x1 - start_x) / scale))
            y1 = max(0, int((y1 - start_y) / scale))
            x2 = min(original_w, int((x2 - start_x) / scale))
            y2 = min(original_h, int((y2 - start_y) / scale))
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            class_id = class_ids[i]
            confidence = max_class_probs[i]
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'class_id': class_id,
                'class_name': class_name,
                'confidence': float(confidence)
            })
        
        return results
    
    def _sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def _nms(self, boxes, scores, threshold):
        """非极大值抑制"""
        if len(boxes) == 0:
            return []
            
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            other_indices = indices[1:]
            
            xx1 = np.maximum(x1[current], x1[other_indices])
            yy1 = np.maximum(y1[current], y1[other_indices])
            xx2 = np.minimum(x2[current], x2[other_indices])
            yy2 = np.minimum(y2[current], y2[other_indices])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[current] + areas[other_indices] - intersection + 1e-6)
            indices = other_indices[iou <= threshold]
        
        return keep
    
    def detect_from_image(self, image: np.ndarray, 
                         conf_threshold: float = 0.6,
                         nms_threshold: float = 0.3,
                         return_details: bool = False) -> Dict[str, Any]:
        """
        从图像检测动物
        
        Args:
            image: 输入图像 (BGR格式)
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            return_details: 是否返回详细信息
            
        Returns:
            检测结果字典
        """
        start_time = time.time()
        
        # 预处理
        input_data, scale, offset = self.preprocess_image(image)
        
        # 推理
        predictions = self.inference(input_data)
        
        # 后处理
        detections = self.postprocess(
            predictions, image.shape[:2], scale, offset,
            conf_threshold, nms_threshold
        )
        
        # 统计动物数量
        animal_counts = {}
        for class_name in self.class_names.values():
            animal_counts[class_name] = 0
            
        for detection in detections:
            class_name = detection['class_name']
            animal_counts[class_name] += 1
        
        # 计算处理时间
        process_time = time.time() - start_time
        self.total_inferences += 1
        self.total_time += process_time
        
        result = {
            'animal_counts': animal_counts,
            'total_animals': sum(animal_counts.values()),
            'process_time_ms': round(process_time * 1000, 2)
        }
        
        if return_details:
            result['detections'] = detections
            result['image_shape'] = image.shape[:2]
            result['confidence_threshold'] = conf_threshold
            result['nms_threshold'] = nms_threshold
        
        return result
    
    def detect_from_file(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """
        从图像文件检测动物
        
        Args:
            image_path: 图像文件路径
            **kwargs: 传递给detect_from_image的参数
            
        Returns:
            检测结果字典
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像文件: {image_path}")
            
        result = self.detect_from_image(image, **kwargs)
        result['image_path'] = image_path
        
        return result
    
    def detect_from_base64(self, base64_string: str, **kwargs) -> Dict[str, Any]:
        """
        从base64编码的图像检测动物
        
        Args:
            base64_string: base64编码的图像字符串
            **kwargs: 传递给detect_from_image的参数
            
        Returns:
            检测结果字典
        """
        try:
            # 解码base64
            image_data = base64.b64decode(base64_string)
            image_pil = Image.open(BytesIO(image_data))
            
            # 转换为OpenCV格式
            image_rgb = np.array(image_pil)
            if len(image_rgb.shape) == 3:
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            else:
                image = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
                
            return self.detect_from_image(image, **kwargs)
            
        except Exception as e:
            raise ValueError(f"无法解析base64图像: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取API使用统计"""
        avg_time = self.total_time / self.total_inferences if self.total_inferences > 0 else 0
        
        return {
            'total_inferences': self.total_inferences,
            'total_time_seconds': round(self.total_time, 2),
            'average_time_ms': round(avg_time * 1000, 2),
            'average_fps': round(1 / avg_time if avg_time > 0 else 0, 2)
        }
    
    def batch_process(self, image_list: List[np.ndarray], 
                     conf_threshold: float = 0.6,
                     nms_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        批量处理图像
        
        Args:
            image_list: 图像列表
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            批量检测结果列表
        """
        results = []
        
        for i, image in enumerate(image_list):
            try:
                result = self.detect_from_image(
                    image, conf_threshold, nms_threshold, return_details=False
                )
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'animal_counts': {name: 0 for name in self.class_names.values()},
                    'total_animals': 0
                })
        
        return results


# 示例使用代码
def example_usage():
    """API使用示例"""
    
    api = TensorRTAnimalDetectionAPI('best9999.engine')    
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
    if ret:
        result2 = api.detect_from_image(frame, return_details=True)
        print(f"实时检测结果: {result2['animal_counts']}")
        print(f"处理时间: {result2['process_time_ms']}ms")
    cap.release()
    


if __name__ == "__main__":
    example_usage()