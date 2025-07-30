#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO模型ONNX转换工具
支持将.pt模型转换为ONNX格式，作为TensorRT转换的备选方案
ONNX模型可以在多种推理引擎上运行，包括ONNX Runtime
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXConverter:
    def __init__(self):
        """初始化ONNX转换器"""
        pass
    
    def convert_pt_to_onnx(self, pt_model_path, output_dir=None, input_size=(640, 640), 
                          batch_size=1, dynamic_batch=False, opset_version=11, simplify=True):
        """
        将PyTorch .pt模型转换为ONNX格式
        
        Args:
            pt_model_path: .pt模型文件路径
            output_dir: 输出目录，默认为模型文件所在目录
            input_size: 输入图像尺寸 (width, height)
            batch_size: 批处理大小
            dynamic_batch: 是否启用动态批处理
            opset_version: ONNX操作集版本
            simplify: 是否简化模型
        
        Returns:
            ONNX模型文件路径，失败返回None
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
        
        # 生成输出文件名
        model_name = pt_path.stem
        dynamic_suffix = "_dynamic" if dynamic_batch else ""
        onnx_path = output_dir / f"{model_name}_{input_size[0]}x{input_size[1]}{dynamic_suffix}.onnx"
        
        logger.info("="*60)
        logger.info("开始YOLO模型ONNX转换")
        logger.info(f"输入模型: {pt_model_path}")
        logger.info(f"输出路径: {onnx_path}")
        logger.info(f"输入尺寸: {input_size}")
        logger.info(f"批处理大小: {batch_size}")
        logger.info(f"动态批处理: {dynamic_batch}")
        logger.info(f"ONNX操作集版本: {opset_version}")
        logger.info(f"模型简化: {simplify}")
        logger.info("="*60)
        
        try:
            # 加载YOLO模型
            logger.info("加载YOLO模型...")
            model = YOLO(pt_model_path)
            
            # 导出为ONNX
            logger.info("开始导出ONNX模型...")
            start_time = time.time()
            
            success = model.export(
                format='onnx',
                imgsz=input_size,
                batch=batch_size,
                device='cpu',  # 使用CPU导出以避免CUDA内存问题
                simplify=simplify,
                opset=opset_version,
                dynamic=dynamic_batch
            )
            
            export_time = time.time() - start_time
            
            if success:
                # 重命名导出的文件
                default_onnx_path = pt_model_path.replace('.pt', '.onnx')
                if os.path.exists(default_onnx_path) and default_onnx_path != str(onnx_path):
                    os.rename(default_onnx_path, str(onnx_path))
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info("="*60)
                logger.info("ONNX转换成功！")
                logger.info(f"输出文件: {onnx_path}")
                logger.info(f"文件大小: {file_size:.2f} MB")
                logger.info(f"转换耗时: {export_time:.2f} 秒")
                logger.info("="*60)
                
                return str(onnx_path)
            else:
                logger.error("ONNX导出失败")
                return None
                
        except Exception as e:
            logger.error(f"ONNX转换失败: {str(e)}")
            return None
    
    def validate_onnx_model(self, onnx_model_path, input_size=(640, 640)):
        """
        验证ONNX模型是否正确导出
        
        Args:
            onnx_model_path: ONNX模型文件路径
            input_size: 输入图像尺寸
        """
        logger.info(f"验证ONNX模型: {onnx_model_path}")
        
        try:
            import onnx
            import onnxruntime as ort
            
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_model_path)
            
            # 检查模型
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX模型结构验证通过")
            
            # 打印模型信息
            logger.info(f"ONNX版本: {onnx_model.opset_import[0].version}")
            logger.info(f"生产者: {onnx_model.producer_name}")
            
            # 打印输入输出信息
            for input_info in onnx_model.graph.input:
                shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
                logger.info(f"输入: {input_info.name}, 形状: {shape}")
            
            for output_info in onnx_model.graph.output:
                shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
                logger.info(f"输出: {output_info.name}, 形状: {shape}")
            
            # 测试ONNX Runtime推理
            logger.info("测试ONNX Runtime推理...")
            
            # 创建推理会话
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
            
            # 获取输入输出名称
            input_name = ort_session.get_inputs()[0].name
            output_names = [output.name for output in ort_session.get_outputs()]
            
            # 创建测试输入
            test_input = np.random.random((1, 3, input_size[1], input_size[0])).astype(np.float32)
            
            # 推理测试
            start_time = time.time()
            outputs = ort_session.run(output_names, {input_name: test_input})
            inference_time = time.time() - start_time
            
            logger.info(f"ONNX Runtime推理测试成功")
            logger.info(f"推理时间: {inference_time*1000:.2f} ms")
            logger.info(f"输出数量: {len(outputs)}")
            for i, output in enumerate(outputs):
                logger.info(f"输出 {i} 形状: {output.shape}")
            
            # 获取使用的执行provider
            used_providers = ort_session.get_providers()
            logger.info(f"使用的执行Provider: {used_providers[0]}")
            
            return True
            
        except ImportError as e:
            logger.warning(f"无法验证ONNX模型，缺少依赖: {str(e)}")
            logger.warning("请安装: pip install onnx onnxruntime-gpu")
            return False
        except Exception as e:
            logger.error(f"ONNX模型验证失败: {str(e)}")
            return False
    
    def benchmark_onnx_model(self, onnx_model_path, input_size=(640, 640), num_runs=100, 
                           use_gpu=True):
        """
        测试ONNX模型性能
        
        Args:
            onnx_model_path: ONNX模型文件路径
            input_size: 输入图像尺寸
            num_runs: 测试运行次数
            use_gpu: 是否使用GPU
        """
        logger.info(f"开始ONNX模型性能测试: {onnx_model_path}")
        
        try:
            import onnxruntime as ort
            
            # 设置执行provider
            if use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            # 创建推理会话
            ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
            
            # 获取输入输出信息
            input_name = ort_session.get_inputs()[0].name
            output_names = [output.name for output in ort_session.get_outputs()]
            
            logger.info(f"使用执行Provider: {ort_session.get_providers()[0]}")
            
            # 创建测试数据
            test_input = np.random.random((1, 3, input_size[1], input_size[0])).astype(np.float32)
            
            # 预热
            logger.info("预热模型...")
            for _ in range(10):
                _ = ort_session.run(output_names, {input_name: test_input})
            
            # 性能测试
            logger.info(f"开始性能测试 ({num_runs} 次运行)...")
            
            times = []
            for i in range(num_runs):
                start_time = time.time()
                _ = ort_session.run(output_names, {input_name: test_input})
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
            logger.info("ONNX模型性能测试结果:")
            logger.info(f"平均推理时间: {avg_time:.2f} ms")
            logger.info(f"最小推理时间: {min_time:.2f} ms")
            logger.info(f"最大推理时间: {max_time:.2f} ms")
            logger.info(f"标准差: {std_time:.2f} ms")
            logger.info(f"平均FPS: {fps:.1f}")
            logger.info("="*50)
            
        except ImportError:
            logger.error("ONNX Runtime未安装，请安装: pip install onnxruntime-gpu")
        except Exception as e:
            logger.error(f"性能测试失败: {str(e)}")
    
    def convert_multiple_models(self, model_dir, output_dir=None, input_sizes=[(640, 640)], 
                               batch_sizes=[1], **kwargs):
        """
        批量转换多个模型
        
        Args:
            model_dir: 包含.pt模型文件的目录
            output_dir: 输出目录
            input_sizes: 输入尺寸列表
            batch_sizes: 批处理大小列表
            **kwargs: 其他转换参数
        """
        model_dir = Path(model_dir)
        if not model_dir.exists():
            logger.error(f"模型目录不存在: {model_dir}")
            return
        
        # 查找所有.pt文件
        pt_files = list(model_dir.glob("*.pt"))
        if not pt_files:
            logger.warning(f"在目录 {model_dir} 中未找到.pt文件")
            return
        
        logger.info(f"找到 {len(pt_files)} 个.pt模型文件")
        
        success_count = 0
        total_conversions = len(pt_files) * len(input_sizes) * len(batch_sizes)
        
        for pt_file in pt_files:
            for input_size in input_sizes:
                for batch_size in batch_sizes:
                    logger.info(f"转换模型: {pt_file.name}, 尺寸: {input_size}, 批处理: {batch_size}")
                    
                    result = self.convert_pt_to_onnx(
                        pt_model_path=str(pt_file),
                        output_dir=output_dir,
                        input_size=input_size,
                        batch_size=batch_size,
                        **kwargs
                    )
                    
                    if result:
                        success_count += 1
                    
                    logger.info("-" * 50)
        
        logger.info("="*60)
        logger.info("批量转换完成")
        logger.info(f"成功转换: {success_count}/{total_conversions}")
        logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(description='YOLO模型ONNX转换工具')
    parser.add_argument('--model', type=str, required=True,
                       help='.pt模型文件路径或包含.pt文件的目录')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录 (默认: 模型文件所在目录)')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640],
                       help='输入图像尺寸 [width height] (默认: 640 640)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='批处理大小 (默认: 1)')
    parser.add_argument('--dynamic-batch', action='store_true',
                       help='启用动态批处理')
    parser.add_argument('--opset-version', type=int, default=11,
                       help='ONNX操作集版本 (默认: 11)')
    parser.add_argument('--no-simplify', action='store_true',
                       help='不简化模型')
    parser.add_argument('--validate', action='store_true',
                       help='转换完成后验证模型')
    parser.add_argument('--benchmark', action='store_true',
                       help='转换完成后进行性能测试')
    parser.add_argument('--benchmark-only', type=str, default=None,
                       help='仅对指定的ONNX模型文件进行性能测试')
    parser.add_argument('--batch-convert', action='store_true',
                       help='批量转换目录中的所有.pt文件')
    parser.add_argument('--multiple-sizes', action='store_true',
                       help='生成多个尺寸的模型 (320, 640, 1280)')
    
    args = parser.parse_args()
    
    # 检查ONNX相关依赖
    try:
        import onnx
        logger.info(f"ONNX版本: {onnx.__version__}")
    except ImportError:
        logger.error("ONNX未安装，请安装: pip install onnx")
        sys.exit(1)
    
    try:
        import onnxruntime as ort
        logger.info(f"ONNX Runtime版本: {ort.__version__}")
        
        # 检查可用的执行provider
        available_providers = ort.get_available_providers()
        logger.info(f"可用的执行Provider: {available_providers}")
    except ImportError:
        logger.warning("ONNX Runtime未安装，无法进行模型验证和性能测试")
        logger.warning("请安装: pip install onnxruntime-gpu")
    
    # 创建转换器
    converter = ONNXConverter()
    
    # 如果只是性能测试
    if args.benchmark_only:
        if not os.path.exists(args.benchmark_only):
            logger.error(f"ONNX文件不存在: {args.benchmark_only}")
            sys.exit(1)
        
        converter.benchmark_onnx_model(args.benchmark_only, tuple(args.input_size))
        return
    
    # 批量转换
    if args.batch_convert:
        input_sizes = [(320, 320), (640, 640), (1280, 1280)] if args.multiple_sizes else [tuple(args.input_size)]
        batch_sizes = [1, args.batch_size] if args.batch_size > 1 else [1]
        
        converter.convert_multiple_models(
            model_dir=args.model,
            output_dir=args.output_dir,
            input_sizes=input_sizes,
            batch_sizes=batch_sizes,
            dynamic_batch=args.dynamic_batch,
            opset_version=args.opset_version,
            simplify=not args.no_simplify
        )
        return
    
    # 单个模型转换
    onnx_model_path = converter.convert_pt_to_onnx(
        pt_model_path=args.model,
        output_dir=args.output_dir,
        input_size=tuple(args.input_size),
        batch_size=args.batch_size,
        dynamic_batch=args.dynamic_batch,
        opset_version=args.opset_version,
        simplify=not args.no_simplify
    )
    
    if onnx_model_path is None:
        logger.error("模型转换失败")
        sys.exit(1)
    
    # 验证模型
    if args.validate:
        converter.validate_onnx_model(onnx_model_path, tuple(args.input_size))
    
    # 性能测试
    if args.benchmark:
        converter.benchmark_onnx_model(onnx_model_path, tuple(args.input_size))

if __name__ == "__main__":
    main()