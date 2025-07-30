#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO模型TensorRT转换工具
支持将.pt模型转换为TensorRT引擎以获得最佳Jetson性能
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
from ultralytics import YOLO

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TensorRTConverter:
    def __init__(self, verbose=False):
        """
        初始化TensorRT转换器
        
        Args:
            verbose: 是否显示详细日志
        """
        self.verbose = verbose
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        
    def pt_to_onnx(self, pt_model_path, onnx_output_path, input_size=(640, 640), batch_size=1):
        """
        将PyTorch .pt模型转换为ONNX格式
        
        Args:
            pt_model_path: .pt模型文件路径
            onnx_output_path: 输出ONNX文件路径
            input_size: 输入图像尺寸 (width, height)
            batch_size: 批处理大小
        """
        logger.info(f"开始转换 {pt_model_path} 到 ONNX 格式...")
        
        try:
            # 加载YOLO模型
            model = YOLO(pt_model_path)
            
            # 导出为ONNX
            success = model.export(
                format='onnx',
                imgsz=input_size,
                batch=batch_size,
                device='cpu',  # 使用CPU导出以避免CUDA内存问题
                simplify=True,
                opset=11,
                dynamic=False
            )
            
            if success:
                # 重命名导出的文件
                default_onnx_path = pt_model_path.replace('.pt', '.onnx')
                if os.path.exists(default_onnx_path) and default_onnx_path != onnx_output_path:
                    os.rename(default_onnx_path, onnx_output_path)
                
                logger.info(f"ONNX转换成功: {onnx_output_path}")
                return onnx_output_path
            else:
                raise Exception("ONNX导出失败")
                
        except Exception as e:
            logger.error(f"ONNX转换失败: {str(e)}")
            return None
    
    def onnx_to_tensorrt(self, onnx_model_path, trt_output_path, precision='fp16', 
                        max_batch_size=1, workspace_size=1<<30):
        """
        将ONNX模型转换为TensorRT引擎
        
        Args:
            onnx_model_path: ONNX模型文件路径
            trt_output_path: 输出TensorRT引擎文件路径
            precision: 精度模式 ('fp32', 'fp16', 'int8')
            max_batch_size: 最大批处理大小
            workspace_size: 工作空间大小(字节)
        """
        logger.info(f"开始转换 {onnx_model_path} 到 TensorRT 引擎...")
        logger.info(f"精度模式: {precision}")
        
        try:
            # 创建TensorRT构建器
            builder = trt.Builder(self.logger)
            config = builder.create_builder_config()
            
            # 设置工作空间大小
            config.max_workspace_size = workspace_size
            
            # 设置精度
            if precision == 'fp16':
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    logger.info("启用FP16精度")
                else:
                    logger.warning("设备不支持FP16，使用FP32")
            elif precision == 'int8':
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    logger.info("启用INT8精度")
                else:
                    logger.warning("设备不支持INT8，使用FP32")
            
            # 创建网络定义
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # 解析ONNX模型
            parser = trt.OnnxParser(network, self.logger)
            
            with open(onnx_model_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    logger.error("ONNX模型解析失败")
                    for error in range(parser.num_errors):
                        logger.error(f"解析错误: {parser.get_error(error)}")
                    return None
            
            logger.info(f"ONNX模型解析成功，网络有 {network.num_inputs} 个输入和 {network.num_outputs} 个输出")
            
            # 打印输入输出信息
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                logger.info(f"输入 {i}: {input_tensor.name}, 形状: {input_tensor.shape}, 类型: {input_tensor.dtype}")
            
            for i in range(network.num_outputs):
                output_tensor = network.get_output(i)
                logger.info(f"输出 {i}: {output_tensor.name}, 形状: {output_tensor.shape}, 类型: {output_tensor.dtype}")
            
            # 构建引擎
            logger.info("开始构建TensorRT引擎，这可能需要几分钟...")
            start_time = time.time()
            
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                logger.error("TensorRT引擎构建失败")
                return None
            
            build_time = time.time() - start_time
            logger.info(f"TensorRT引擎构建完成，耗时: {build_time:.2f}秒")
            
            # 保存引擎文件
            with open(trt_output_path, 'wb') as f:
                f.write(serialized_engine)
            
            logger.info(f"TensorRT引擎已保存到: {trt_output_path}")
            
            # 获取文件大小
            file_size = os.path.getsize(trt_output_path) / (1024 * 1024)  # MB
            logger.info(f"引擎文件大小: {file_size:.2f} MB")
            
            return trt_output_path
            
        except Exception as e:
            logger.error(f"TensorRT转换失败: {str(e)}")
            return None
    
    def convert_pt_to_tensorrt(self, pt_model_path, output_dir=None, input_size=(640, 640), 
                              precision='fp16', batch_size=1, keep_onnx=False):
        """
        一键转换：从.pt直接转换到TensorRT
        
        Args:
            pt_model_path: .pt模型文件路径
            output_dir: 输出目录，默认为模型文件所在目录
            input_size: 输入图像尺寸
            precision: 精度模式
            batch_size: 批处理大小
            keep_onnx: 是否保留中间的ONNX文件
        
        Returns:
            TensorRT引擎文件路径，失败返回None
        """
        pt_path = Path(pt_model_path)
        if not pt_path.exists():
            logger.error(f"模型文件不存在: {pt_model_path}")
            return None
        
        # 设置输出目录
        if output_dir is None:
            output_dir = pt_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        model_name = pt_path.stem
        onnx_path = output_dir / f"{model_name}_{input_size[0]}x{input_size[1]}.onnx"
        trt_path = output_dir / f"{model_name}_{input_size[0]}x{input_size[1]}_{precision}.engine"
        
        logger.info("="*60)
        logger.info("开始YOLO模型TensorRT转换流程")
        logger.info(f"输入模型: {pt_model_path}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"输入尺寸: {input_size}")
        logger.info(f"精度模式: {precision}")
        logger.info(f"批处理大小: {batch_size}")
        logger.info("="*60)
        
        # 步骤1: 转换为ONNX
        logger.info("步骤 1/2: 转换PT模型到ONNX...")
        onnx_result = self.pt_to_onnx(str(pt_path), str(onnx_path), input_size, batch_size)
        
        if onnx_result is None:
            logger.error("ONNX转换失败，中止流程")
            return None
        
        # 步骤2: 转换为TensorRT
        logger.info("步骤 2/2: 转换ONNX到TensorRT引擎...")
        trt_result = self.onnx_to_tensorrt(str(onnx_path), str(trt_path), precision)
        
        if trt_result is None:
            logger.error("TensorRT转换失败")
            return None
        
        # 清理中间文件
        if not keep_onnx and os.path.exists(onnx_path):
            os.remove(onnx_path)
            logger.info(f"已删除中间ONNX文件: {onnx_path}")
        
        logger.info("="*60)
        logger.info("转换完成！")
        logger.info(f"TensorRT引擎: {trt_result}")
        logger.info("="*60)
        
        return str(trt_result)
    
    def benchmark_engine(self, engine_path, input_size=(640, 640), num_runs=100):
        """
        测试TensorRT引擎性能
        
        Args:
            engine_path: TensorRT引擎文件路径
            input_size: 输入图像尺寸
            num_runs: 测试运行次数
        """
        logger.info(f"开始性能测试: {engine_path}")
        
        try:
            # 加载引擎
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(self.logger)
                engine = runtime.deserialize_cuda_engine(f.read())
            
            if engine is None:
                logger.error("无法加载TensorRT引擎")
                return
            
            # 创建执行上下文
            context = engine.create_execution_context()
            
            # 获取输入输出信息
            input_shape = (1, 3, input_size[1], input_size[0])  # NCHW格式
            input_size_bytes = np.prod(input_shape) * np.dtype(np.float32).itemsize
            
            # 分配GPU内存
            cuda_input = cuda.mem_alloc(input_size_bytes)
            
            # 创建测试数据
            test_data = np.random.random(input_shape).astype(np.float32)
            
            # 预热
            logger.info("预热引擎...")
            for _ in range(10):
                cuda.memcpy_htod(cuda_input, test_data)
                context.execute_v2([int(cuda_input)])
            
            # 性能测试
            logger.info(f"开始性能测试 ({num_runs} 次运行)...")
            
            times = []
            for i in range(num_runs):
                start_time = time.time()
                
                cuda.memcpy_htod(cuda_input, test_data)
                context.execute_v2([int(cuda_input)])
                cuda.Context.synchronize()
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # 转换为毫秒
                
                if (i + 1) % 20 == 0:
                    logger.info(f"已完成 {i + 1}/{num_runs} 次测试")
            
            # 计算统计信息
            times = np.array(times)
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            fps = 1000.0 / avg_time
            
            logger.info("="*50)
            logger.info("性能测试结果:")
            logger.info(f"平均推理时间: {avg_time:.2f} ms")
            logger.info(f"最小推理时间: {min_time:.2f} ms")
            logger.info(f"最大推理时间: {max_time:.2f} ms")
            logger.info(f"标准差: {std_time:.2f} ms")
            logger.info(f"平均FPS: {fps:.1f}")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"性能测试失败: {str(e)}")
        finally:
            # 清理资源
            if 'cuda_input' in locals():
                cuda_input.free()

def main():
    parser = argparse.ArgumentParser(description='YOLO模型TensorRT转换工具')
    parser.add_argument('--model', type=str, required=True,
                       help='.pt模型文件路径')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录 (默认: 模型文件所在目录)')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640],
                       help='输入图像尺寸 [width height] (默认: 640 640)')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'], 
                       default='fp16', help='精度模式 (默认: fp16)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='批处理大小 (默认: 1)')
    parser.add_argument('--keep-onnx', action='store_true',
                       help='保留中间ONNX文件')
    parser.add_argument('--benchmark', action='store_true',
                       help='转换完成后进行性能测试')
    parser.add_argument('--benchmark-only', type=str, default=None,
                       help='仅对指定的TensorRT引擎文件进行性能测试')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细日志')
    
    args = parser.parse_args()
    
    # 检查TensorRT是否可用
    try:
        import tensorrt as trt
        logger.info(f"TensorRT版本: {trt.__version__}")
    except ImportError:
        logger.error("TensorRT未安装，请先安装TensorRT")
        sys.exit(1)
    
    # 检查CUDA是否可用
    try:
        import pycuda.driver as cuda
        cuda.init()
        logger.info(f"CUDA设备数量: {cuda.Device.count()}")
        if cuda.Device.count() > 0:
            device = cuda.Device(0)
            logger.info(f"使用GPU: {device.name()}")
    except Exception as e:
        logger.error(f"CUDA初始化失败: {str(e)}")
        sys.exit(1)
    
    # 创建转换器
    converter = TensorRTConverter(verbose=args.verbose)
    
    # 如果只是性能测试
    if args.benchmark_only:
        if not os.path.exists(args.benchmark_only):
            logger.error(f"引擎文件不存在: {args.benchmark_only}")
            sys.exit(1)
        
        converter.benchmark_engine(args.benchmark_only, tuple(args.input_size))
        return
    
    # 转换模型
    trt_engine_path = converter.convert_pt_to_tensorrt(
        pt_model_path=args.model,
        output_dir=args.output_dir,
        input_size=tuple(args.input_size),
        precision=args.precision,
        batch_size=args.batch_size,
        keep_onnx=args.keep_onnx
    )
    
    if trt_engine_path is None:
        logger.error("模型转换失败")
        sys.exit(1)
    
    # 性能测试
    if args.benchmark:
        converter.benchmark_engine(trt_engine_path, tuple(args.input_size))

if __name__ == "__main__":
    main()