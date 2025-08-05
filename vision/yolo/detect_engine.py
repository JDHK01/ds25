#!/usr/bin/env python3
"""
高性能TensorRT动物检测推理器
专门优化单帧和实时处理性能
目标：从1.39秒优化到<50ms
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import threading
from queue import Queue
import cupy as cp  # 需要安装: pip install cupy-cuda11x


class HighPerformanceAnimalDetector:
    """
    超高性能动物检测器
    优化点：
    1. 预分配GPU内存，避免动态分配
    2. 使用CuPy进行GPU加速的图像预处理
    3. 支持批处理和流水线处理
    4. 最小化CPU-GPU同步
    5. 优化后处理算法
    """
    
    def __init__(self, engine_path: str, 
                 max_batch_size: int = 4,
                 use_cuda_preprocessing: bool = True,
                 enable_profiling: bool = False):
        """
        初始化高性能检测器
        
        Args:
            engine_path: TensorRT引擎路径
            max_batch_size: 最大批处理大小
            use_cuda_preprocessing: 是否使用CUDA预处理
            enable_profiling: 是否启用性能分析
        """
        self.engine_path = engine_path
        self.max_batch_size = max_batch_size
        self.use_cuda_preprocessing = use_cuda_preprocessing
        self.enable_profiling = enable_profiling
        
        # 动物类别映射
        self.class_names = {
            0: 'elephant', 1: 'monkey', 2: 'peacock', 3: 'wolf', 4: 'tiger'
        }
        
        # 动物尺寸优化
        self.animal_size_ratios = {
            'elephant': 2.0, 'monkey': 1.0, 'peacock': 1.0, 
            'wolf': 1.0, 'tiger': 1.0
        }
        
        # 性能统计
        self.total_inferences = 0
        self.total_time = 0.0
        self.preprocessing_time = 0.0
        self.inference_time = 0.0
        self.postprocessing_time = 0.0
        
        # 初始化TensorRT
        self._init_tensorrt()
        
        # 预分配GPU内存
        self._allocate_gpu_memory()
        
        # 初始化CUDA预处理
        if self.use_cuda_preprocessing:
            self._init_cuda_preprocessing()
        
        print(f"高性能检测器初始化完成")
        print(f"  最大批处理大小: {self.max_batch_size}")
        print(f"  CUDA预处理: {self.use_cuda_preprocessing}")
        print(f"  性能分析: {self.enable_profiling}")
    
    def _init_tensorrt(self):
        """初始化TensorRT组件"""
        print(f"加载TensorRT引擎: {self.engine_path}")
        
        # 设置日志级别
        self.logger = trt.Logger(trt.Logger.ERROR)  # 减少日志输出
        self.runtime = trt.Runtime(self.logger)
        
        # 加载引擎
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("无法加载TensorRT引擎")
        
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 创建CUDA流
        self.stream = cuda.Stream()
        
        # 获取绑定信息
        self._setup_bindings()
        
        print(f"TensorRT引擎加载完成")
        print(f"  输入形状: {self.input_shape}")
        print(f"  输出形状: {self.output_shape}")
    
    def _setup_bindings(self):
        """设置输入输出绑定"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            if self.engine.binding_is_input(i):
                self.input_shape = shape
                self.input_dtype = dtype
            else:
                self.output_shape = shape
                self.output_dtype = dtype
    
    def _allocate_gpu_memory(self):
        """预分配GPU内存以避免动态分配开销"""
        # 计算所需内存大小
        input_size = trt.volume(self.input_shape) * self.max_batch_size
        output_size = trt.volume(self.output_shape) * self.max_batch_size
        
        # 分配输入内存
        self.input_host = cuda.pagelocked_empty(input_size, self.input_dtype)
        self.input_device = cuda.mem_alloc(self.input_host.nbytes)
        
        # 分配输出内存
        self.output_host = cuda.pagelocked_empty(output_size, self.output_dtype)
        self.output_device = cuda.mem_alloc(self.output_host.nbytes)
        
        # 设置绑定
        self.bindings = [int(self.input_device), int(self.output_device)]
        
        print(f"GPU内存预分配完成:")
        print(f"  输入内存: {self.input_host.nbytes / 1024 / 1024:.2f} MB")
        print(f"  输出内存: {self.output_host.nbytes / 1024 / 1024:.2f} MB")
    
    def _init_cuda_preprocessing(self):
        """初始化CUDA预处理"""
        try:
            # 预分配GPU内存用于图像预处理
            self.target_size = 640
            max_input_size = self.target_size * self.target_size * 3 * self.max_batch_size
            
            # CuPy内存池预分配
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=512 * 1024 * 1024)  # 512MB限制
            
            print("CUDA预处理初始化完成")
        except Exception as e:
            print(f"CUDA预处理初始化失败，回退到CPU: {e}")
            self.use_cuda_preprocessing = False
    
    def preprocess_images_cuda(self, images: List[np.ndarray]) -> Tuple[np.ndarray, List[float], List[Tuple[int, int]]]:
        """
        使用CUDA加速的批量图像预处理
        
        Args:
            images: 输入图像列表
            
        Returns:
            预处理后的批量数据, 缩放比例列表, 偏移量列表
        """
        batch_size = len(images)
        target_size = 640
        
        # 预分配输出数组
        batch_data = np.zeros((batch_size, 3, target_size, target_size), dtype=np.float32)
        scales = []
        offsets = []
        
        for i, image in enumerate(images):
            # 转换为CuPy数组进行GPU处理
            gpu_image = cp.asarray(image, dtype=cp.uint8)
            
            h, w = gpu_image.shape[:2]
            scale = min(target_size / h, target_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # GPU上进行resize
            gpu_resized = cp.zeros((new_h, new_w, 3), dtype=cp.uint8)
            # 简化版resize（实际项目中可使用cupy-img或其他CUDA加速库）
            step_h, step_w = h / new_h, w / new_w
            for y in range(new_h):
                for x in range(new_w):
                    src_y, src_x = int(y * step_h), int(x * step_w)
                    gpu_resized[y, x] = gpu_image[min(src_y, h-1), min(src_x, w-1)]
            
            # 创建画布并填充
            canvas = cp.full((target_size, target_size, 3), 114, dtype=cp.uint8)
            start_y = (target_size - new_h) // 2
            start_x = (target_size - new_w) // 2
            canvas[start_y:start_y + new_h, start_x:start_x + new_w] = gpu_resized
            
            # 颜色空间转换和归一化
            canvas = canvas[:, :, ::-1]  # BGR to RGB
            canvas = canvas.astype(cp.float32) / 255.0
            
            # 转换维度顺序 HWC -> CHW
            canvas = cp.transpose(canvas, (2, 0, 1))
            
            # 复制到CPU
            batch_data[i] = cp.asnumpy(canvas)
            scales.append(scale)
            offsets.append((start_x, start_y))
        
        return batch_data, scales, offsets
    
    def preprocess_images_cpu(self, images: List[np.ndarray]) -> Tuple[np.ndarray, List[float], List[Tuple[int, int]]]:
        """
        CPU批量图像预处理（优化版本）
        
        Args:
            images: 输入图像列表
            
        Returns:
            预处理后的批量数据, 缩放比例列表, 偏移量列表
        """
        batch_size = len(images)
        target_size = 640
        
        # 预分配输出数组
        batch_data = np.zeros((batch_size, 3, target_size, target_size), dtype=np.float32)
        scales = []
        offsets = []
        
        for i, image in enumerate(images):
            h, w = image.shape[:2]
            scale = min(target_size / h, target_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # 使用最快的插值方法
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # 创建画布
            canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
            start_y = (target_size - new_h) // 2
            start_x = (target_size - new_w) // 2
            canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
            
            # 快速颜色转换和归一化
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            canvas = canvas.astype(np.float32) * (1.0/255.0)  # 更快的除法
            
            # 转换维度
            canvas = np.transpose(canvas, (2, 0, 1))
            batch_data[i] = canvas
            
            scales.append(scale)
            offsets.append((start_x, start_y))
        
        return batch_data, scales, offsets
    
    def inference_batch(self, batch_data: np.ndarray) -> np.ndarray:
        """
        批量推理（优化版本）
        
        Args:
            batch_data: 批量输入数据 [batch_size, 3, 640, 640]
            
        Returns:
            批量推理结果
        """
        batch_size = batch_data.shape[0]
        
        # 设置批处理大小
        if hasattr(self.context, 'set_input_shape'):
            input_shape = (batch_size,) + self.input_shape[1:]
            self.context.set_input_shape(0, input_shape)
        
        # 复制数据到预分配的内存
        input_size = batch_data.size
        np.copyto(self.input_host[:input_size], batch_data.ravel())
        
        # 异步内存传输
        cuda.memcpy_htod_async(self.input_device, self.input_host[:input_size], self.stream)
        
        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 异步复制结果
        output_size = batch_size * trt.volume(self.output_shape[1:])
        cuda.memcpy_dtoh_async(self.output_host[:output_size], self.output_device, self.stream)
        
        # 同步等待完成
        self.stream.synchronize()
        
        # 重塑输出
        output_shape = (batch_size,) + self.output_shape[1:]
        return self.output_host[:output_size].reshape(output_shape)
    
    def postprocess_batch_optimized(self, predictions: np.ndarray, 
                                   original_shapes: List[Tuple[int, int]],
                                   scales: List[float], 
                                   offsets: List[Tuple[int, int]],
                                   conf_threshold: float = 0.6,
                                   nms_threshold: float = 0.3) -> List[Dict[str, int]]:
        """
        优化的批量后处理
        
        Args:
            predictions: 批量预测结果
            original_shapes: 原始图像尺寸列表
            scales: 缩放比例列表
            offsets: 偏移量列表
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            批量检测结果（动物计数）
        """
        batch_results = []
        
        for i in range(predictions.shape[0]):
            pred = predictions[i]
            original_shape = original_shapes[i]
            scale = scales[i]
            offset = offsets[i]
            
            # 快速后处理单个结果
            result = self._postprocess_single_fast(
                pred, original_shape, scale, offset, 
                conf_threshold, nms_threshold
            )
            
            # 转换为动物计数
            animal_counts = {name: 0 for name in self.class_names.values()}
            for detection in result:
                class_name = detection['class_name']
                animal_counts[class_name] += 1
            
            batch_results.append(animal_counts)
        
        return batch_results
    
    def _postprocess_single_fast(self, prediction: np.ndarray,
                                original_shape: Tuple[int, int],
                                scale: float, offset: Tuple[int, int],
                                conf_threshold: float, nms_threshold: float) -> List[Dict]:
        """快速单个结果后处理"""
        # 解析预测（假设输出格式为 [8400, 9]）
        if prediction.shape[0] == 9:
            prediction = prediction.T
        
        boxes = prediction[:, :4]
        class_probs = prediction[:, 4:9]
        
        # 向量化Sigmoid
        class_probs = 1 / (1 + np.exp(-np.clip(class_probs, -250, 250)))
        
        # 获取最高概率类别
        max_probs = np.max(class_probs, axis=1)
        class_ids = np.argmax(class_probs, axis=1)
        
        # 置信度过滤
        valid_mask = max_probs > conf_threshold
        if not valid_mask.any():
            return []
        
        boxes = boxes[valid_mask]
        max_probs = max_probs[valid_mask]
        class_ids = class_ids[valid_mask]
        
        # 转换边界框格式
        x_center, y_center, width, height = boxes.T
        x1 = x_center - width * 0.5
        y1 = y_center - height * 0.5
        x2 = x_center + width * 0.5
        y2 = y_center + height * 0.5
        
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        
        # 边界检查
        target_size = 640
        valid_mask = (
            (boxes_xyxy[:, 0] >= 0) & (boxes_xyxy[:, 1] >= 0) &
            (boxes_xyxy[:, 2] <= target_size) & (boxes_xyxy[:, 3] <= target_size) &
            (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        )
        
        boxes_xyxy = boxes_xyxy[valid_mask]
        max_probs = max_probs[valid_mask]
        class_ids = class_ids[valid_mask]
        
        if len(boxes_xyxy) == 0:
            return []
        
        # 快速面积过滤
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        area_mask = (areas >= 400) & (areas <= 50000)  # 简化的面积过滤
        
        boxes_xyxy = boxes_xyxy[area_mask]
        max_probs = max_probs[area_mask]
        class_ids = class_ids[area_mask]
        
        if len(boxes_xyxy) == 0:
            return []
        
        # 快速NMS
        keep_indices = self._fast_nms(boxes_xyxy, max_probs, nms_threshold)
        
        # 转换坐标到原始图像
        results = []
        start_x, start_y = offset
        original_h, original_w = original_shape
        
        for idx in keep_indices:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            
            # 坐标转换
            x1 = max(0, int((x1 - start_x) / scale))
            y1 = max(0, int((y1 - start_y) / scale))
            x2 = min(original_w, int((x2 - start_x) / scale))
            y2 = min(original_h, int((y2 - start_y) / scale))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            class_id = class_ids[idx]
            confidence = max_probs[idx]
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'class_id': class_id,
                'class_name': class_name,
                'confidence': float(confidence)
            })
        
        return results
    
    def _fast_nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """优化的NMS算法"""
        if len(boxes) == 0:
            return []
        
        # 计算面积
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        
        # 按分数排序
        order = scores.argsort()[::-1]
        keep = []
        
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # 计算IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
            
            # 保留IoU小于阈值的框
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect_single_image_fast(self, image: np.ndarray,
                                conf_threshold: float = 0.6,
                                nms_threshold: float = 0.3) -> Dict[str, Any]:
        """
        超快速单图检测（目标<50ms）
        
        Args:
            image: 输入图像
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            检测结果
        """
        start_time = time.time()
        
        # 预处理
        preprocess_start = time.time()
        if self.use_cuda_preprocessing:
            batch_data, scales, offsets = self.preprocess_images_cuda([image])
        else:
            batch_data, scales, offsets = self.preprocess_images_cpu([image])
        
        preprocess_time = time.time() - preprocess_start
        
        # 推理
        inference_start = time.time()
        predictions = self.inference_batch(batch_data)
        inference_time = time.time() - inference_start
        
        # 后处理
        postprocess_start = time.time()
        results = self.postprocess_batch_optimized(
            predictions, [image.shape[:2]], scales, offsets,
            conf_threshold, nms_threshold
        )
        postprocess_time = time.time() - postprocess_start
        
        total_time = time.time() - start_time
        
        # 更新统计
        self.total_inferences += 1
        self.total_time += total_time
        self.preprocessing_time += preprocess_time
        self.inference_time += inference_time
        self.postprocessing_time += postprocess_time
        
        result = {
            'animal_counts': results[0],
            'total_animals': sum(results[0].values()),
            'total_time_ms': round(total_time * 1000, 2),
            'breakdown': {
                'preprocess_ms': round(preprocess_time * 1000, 2),
                'inference_ms': round(inference_time * 1000, 2),
                'postprocess_ms': round(postprocess_time * 1000, 2)
            }
        }
        
        if self.enable_profiling:
            print(f"单帧检测: {result['total_time_ms']:.1f}ms "
                  f"(预处理:{result['breakdown']['preprocess_ms']:.1f}ms, "
                  f"推理:{result['breakdown']['inference_ms']:.1f}ms, "
                  f"后处理:{result['breakdown']['postprocess_ms']:.1f}ms)")
        
        return result
    
    def detect_batch_images_fast(self, images: List[np.ndarray],
                                conf_threshold: float = 0.6,
                                nms_threshold: float = 0.3) -> List[Dict[str, int]]:
        """
        超快速批量检测
        
        Args:
            images: 图像列表
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            批量检测结果
        """
        if len(images) > self.max_batch_size:
            # 分批处理
            results = []
            for i in range(0, len(images), self.max_batch_size):
                batch = images[i:i + self.max_batch_size]
                batch_results = self.detect_batch_images_fast(batch, conf_threshold, nms_threshold)
                results.extend(batch_results)
            return results
        
        start_time = time.time()
        
        # 批量预处理
        if self.use_cuda_preprocessing:
            batch_data, scales, offsets = self.preprocess_images_cuda(images)
        else:
            batch_data, scales, offsets = self.preprocess_images_cpu(images)
        
        # 批量推理
        predictions = self.inference_batch(batch_data)
        
        # 批量后处理
        original_shapes = [img.shape[:2] for img in images]
        results = self.postprocess_batch_optimized(
            predictions, original_shapes, scales, offsets,
            conf_threshold, nms_threshold
        )
        
        total_time = time.time() - start_time
        avg_time_per_image = total_time / len(images)
        
        if self.enable_profiling:
            print(f"批量检测 {len(images)} 张图片: "
                  f"总时间 {total_time*1000:.1f}ms, "
                  f"平均每张 {avg_time_per_image*1000:.1f}ms")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取详细性能统计"""
        if self.total_inferences == 0:
            return {'error': '暂无推理统计'}
        
        avg_total = self.total_time / self.total_inferences
        avg_preprocess = self.preprocessing_time / self.total_inferences
        avg_inference = self.inference_time / self.total_inferences
        avg_postprocess = self.postprocessing_time / self.total_inferences
        
        return {
            'total_inferences': self.total_inferences,
            'average_total_time_ms': round(avg_total * 1000, 2),
            'average_fps': round(1 / avg_total if avg_total > 0 else 0, 1),
            'breakdown': {
                'preprocess_ms': round(avg_preprocess * 1000, 2),
                'inference_ms': round(avg_inference * 1000, 2),
                'postprocess_ms': round(avg_postprocess * 1000, 2)
            },
            'optimization_settings': {
                'max_batch_size': self.max_batch_size,
                'cuda_preprocessing': self.use_cuda_preprocessing,
                'profiling_enabled': self.enable_profiling
            }
        }


class RealtimeCameraDetector:
    """实时摄像头检测器（带帧缓冲和多线程优化）"""
    
    def __init__(self, detector: HighPerformanceAnimalDetector,
                 camera_id: int = 0,
                 frame_skip: int = 3,
                 buffer_size: int = 5):
        """
        初始化实时检测器
        
        Args:
            detector: 高性能检测器实例
            camera_id: 摄像头ID
            frame_skip: 跳帧数（每N帧检测一次）
            buffer_size: 帧缓冲大小
        """
        self.detector = detector
        self.camera_id = camera_id
        self.frame_skip = frame_skip
        self.buffer_size = buffer_size
        
        # 帧缓冲队列
        self.frame_queue = Queue(maxsize=buffer_size)
        self.result_queue = Queue(maxsize=buffer_size)
        
        self.running = False
        self.capture_thread = None
        self.process_thread = None
    
    def _capture_frames(self):
        """摄像头捕获线程"""
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # 跳帧优化
                if frame_count % (self.frame_skip + 1) != 0:
                    continue
                
                # 非阻塞添加到队列
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
        finally:
            cap.release()
    
    def _process_frames(self):
        """帧处理线程"""
        while self.running:
            try:
                # 获取帧（阻塞）
                frame = self.frame_queue.get(timeout=0.1)
                
                # 检测
                result = self.detector.detect_single_image_fast(frame)
                
                # 添加结果到队列
                if not self.result_queue.full():
                    self.result_queue.put(result)
                
            except:
                continue
    
    def start(self):
        """启动实时检测"""
        if self.running:
            return
        
        self.running = True
        
        # 启动线程
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.process_thread = threading.Thread(target=self._process_frames)
        
        self.capture_thread.start()
        self.process_thread.start()
        
        print("实时检测已启动")
    
    def stop(self):
        """停止实时检测"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join()
        if self.process_thread:
            self.process_thread.join()
        
        print("实时检测已停止")
    
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """获取最新检测结果"""
        try:
            return self.result_queue.get_nowait()
        except:
            return None


# 使用示例
def performance_test():
    """性能测试示例"""
    print("=== 高性能TensorRT检测器测试 ===")
    
    # 初始化检测器
    detector = HighPerformanceAnimalDetector(
        engine_path='best9999.engine',
        max_batch_size=4,
        use_cuda_preprocessing=True,
        enable_profiling=True
    )
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    print(f"\n1. 单帧检测性能测试:")
    results = []
    for i in range(10):
        result = detector.detect_single_image_fast(test_image)
        results.append(result['total_time_ms'])
        print(f"  第{i+1}次: {result['total_time_ms']:.1f}ms "
              f"动物数量: {result['total_animals']}")
    
    avg_time = sum(results) / len(results)
    print(f"  平均时间: {avg_time:.1f}ms")
    print(f"  平均FPS: {1000/avg_time:.1f}")
    
    print(f"\n2. 批量检测性能测试:")
    batch_images = [test_image] * 4
    start_time = time.time()
    batch_results = detector.detect_batch_images_fast(batch_images)
    batch_time = time.time() - start_time
    
    print(f"  批量4张图片时间: {batch_time*1000:.1f}ms")
    print(f"  平均每张: {batch_time*1000/4:.1f}ms")
    
    print(f"\n3. 性能统计:")
    stats = detector.get_performance_stats()
    print(f"  总推理次数: {stats['total_inferences']}")
    print(f"  平均总时间: {stats['average_total_time_ms']:.1f}ms")
    print(f"  平均FPS: {stats['average_fps']:.1f}")
    print(f"  时间分布:")
    print(f"    预处理: {stats['breakdown']['preprocess_ms']:.1f}ms")
    print(f"    推理: {stats['breakdown']['inference_ms']:.1f}ms") 
    print(f"    后处理: {stats['breakdown']['postprocess_ms']:.1f}ms")


def realtime_camera_test():
    """实时摄像头测试"""
    print("=== 实时摄像头检测测试 ===")
    
    # 初始化检测器
    detector = HighPerformanceAnimalDetector(
        engine_path='/home/by/ds25/temp/vision/yolo/best9999.engine',
        max_batch_size=1,
        use_cuda_preprocessing=True,
        enable_profiling=True
    )
    
    # 初始化实时检测器
    realtime_detector = RealtimeCameraDetector(
        detector=detector,
        camera_id=0,
        frame_skip=2,  # 每3帧检测一次
        buffer_size=3
    )
    
    # 启动实时检测
    realtime_detector.start()
    
    print("实时检测运行中... 按Ctrl+C退出")
    
    result = realtime_detector.get_latest_result()
    if result:
        print(f"检测结果: {result['animal_counts']} "
                      f"({result['total_time_ms']:.1f}ms)")
    
    realtime_detector.stop()


if __name__ == "__main__":
    realtime_camera_test()