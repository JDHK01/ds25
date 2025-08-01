import cv2
import numpy as np
import onnxruntime as ort
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
from collections import Counter
import json
import sys


class YOLOv8AnimalDetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.5):
        """初始化YOLOv8 ONNX动物检测器"""
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 加载ONNX模型
        self.session = ort.InferenceSession(model_path)
        
        # 获取模型输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 获取输入尺寸
        input_shape = self.session.get_inputs()[0].shape
        print(f"原始输入形状: {input_shape}")
        
        # 处理动态尺寸
        if isinstance(input_shape[2], str) or input_shape[2] == -1:
            self.input_height = 640
        else:
            self.input_height = int(input_shape[2])
            
        if isinstance(input_shape[3], str) or input_shape[3] == -1:
            self.input_width = 640
        else:
            self.input_width = int(input_shape[3])
        
        print(f"模型输入尺寸: {self.input_width}x{self.input_height}")
        
        # 动物类别名称
        self.class_names = ['elephant', 'monkey', 'peacock', 'wolf', 'tiger']
        
    def preprocess(self, image):
        """预处理输入图像"""
        input_image = cv2.resize(image, (self.input_width, self.input_height))
        input_image = input_image.astype(np.float32) / 255.0
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)
        return input_image
    
    def postprocess(self, outputs, original_shape):
        """后处理模型输出"""
        predictions = outputs[0][0].T  # (8400, 84)
        
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # 过滤低置信度
        valid_detections = confidences > self.conf_threshold
        boxes = boxes[valid_detections]
        confidences = confidences[valid_detections]
        class_ids = class_ids[valid_detections]
        
        if len(boxes) == 0:
            return [], [], []
        
        # 坐标转换
        original_height, original_width = original_shape[:2]
        scale_x = original_width / self.input_width
        scale_y = original_height / self.input_height
        
        x_center = boxes[:, 0] * scale_x
        y_center = boxes[:, 1] * scale_y
        width = boxes[:, 2] * scale_x
        height = boxes[:, 3] * scale_y
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        boxes = np.column_stack([x1, y1, x2, y2])
        
        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return boxes[indices], confidences[indices], class_ids[indices]
        else:
            return [], [], []
    
    def detect_animals(self, image, show_result=False):
        """
        检测图像中的动物并返回统计结果
        
        Args:
            image: 输入图像 (BGR格式)
            show_result: 是否显示可视化结果
            
        Returns:
            dict: 动物统计结果，格式如 {'elephant': 4, 'monkey': 3, ...}
        """
        # 执行检测
        input_image = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_image})
        boxes, confidences, class_ids = self.postprocess(outputs, image.shape)
        
        # 统计动物数量
        animal_counts = self._count_animals(class_ids)
        
        # 如果需要显示结果
        if show_result and len(boxes) > 0:
            result_image = self._draw_detections(image.copy(), boxes, confidences, class_ids)
            cv2.imshow('Detection Result', result_image)
            cv2.waitKey(1)
        
        return animal_counts
    
    def _count_animals(self, class_ids):
        """统计动物数量"""
        animal_counts = {}
        
        # 初始化所有动物类别为0
        for class_name in self.class_names:
            animal_counts[class_name] = 0
        
        # 统计检测到的动物
        if len(class_ids) > 0:
            for class_id in class_ids:
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    animal_counts[class_name] += 1
        
        # 只返回数量大于0的动物
        return {k: v for k, v in animal_counts.items() if v > 0}
    
    def _draw_detections(self, image, boxes, confidences, class_ids):
        """在图像上绘制检测结果"""
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            confidence = confidences[i]
            class_id = int(class_ids[i])
            
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"Class{class_id}"
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image
    
    def detect_from_camera(self, camera_id=0, show_result=True, duration=None):
        """
        从摄像头检测动物
        
        Args:
            camera_id: 摄像头ID
            show_result: 是否显示可视化结果
            duration: 检测持续时间(秒)，None表示持续到按q键退出
            
        Returns:
            dict: 动物统计结果
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头 {camera_id}")
        
        print("开始摄像头检测，按 'q' 键退出...")
        
        all_detections = []
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测动物
                animal_counts = self.detect_animals(frame, show_result=show_result)
                
                # 收集所有检测结果
                for animal, count in animal_counts.items():
                    all_detections.extend([animal] * count)
                
                # 显示统计信息
                if animal_counts:
                    print(f"当前帧检测结果: {animal_counts}")
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if duration and (time.time() - start_time) > duration:
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # 统计总体结果
        total_counts = dict(Counter(all_detections))
        return total_counts
    
    def detect_from_image(self, image_path, show_result=True, save_result=None):
        """
        从图像文件检测动物
        
        Args:
            image_path: 图像文件路径
            show_result: 是否显示可视化结果
            save_result: 保存结果图像的路径
            
        Returns:
            dict: 动物统计结果
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        animal_counts = self.detect_animals(image, show_result=False)
        
        if show_result or save_result:
            # 绘制检测结果
            input_image = self.preprocess(image)
            outputs = self.session.run(self.output_names, {self.input_name: input_image})
            boxes, confidences, class_ids = self.postprocess(outputs, image.shape)
            
            if len(boxes) > 0:
                result_image = self._draw_detections(image.copy(), boxes, confidences, class_ids)
                
                if show_result:
                    cv2.imshow('Detection Result', result_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                if save_result:
                    cv2.imwrite(save_result, result_image)
                    print(f"结果图像已保存至: {save_result}")
        
        return animal_counts

# GUI版本(可选)
class AnimalDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("动物检测器")
        self.root.geometry("1000x800")
        
        self.detector = None
        self.is_detecting = False
        self.cap = None
        self.show_result = True
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 模型加载
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(model_frame, text="ONNX模型:").pack(side=tk.LEFT)
        self.model_path_var = tk.StringVar()
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=40)
        model_entry.pack(side=tk.LEFT, padx=(5, 5))
        ttk.Button(model_frame, text="浏览", command=self.browse_model).pack(side=tk.LEFT)
        ttk.Button(model_frame, text="加载模型", command=self.load_model).pack(side=tk.LEFT, padx=(5, 0))
        
        # 参数设置
        param_frame = ttk.Frame(control_frame)
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(param_frame, text="置信度:").pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=0.5)
        ttk.Scale(param_frame, from_=0.1, to=1.0, variable=self.conf_var, 
                 orient=tk.HORIZONTAL, length=100).pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(param_frame, text="NMS:").pack(side=tk.LEFT)
        self.iou_var = tk.DoubleVar(value=0.5)
        ttk.Scale(param_frame, from_=0.1, to=1.0, variable=self.iou_var, 
                 orient=tk.HORIZONTAL, length=100).pack(side=tk.LEFT, padx=(5, 10))
        
        # 显示选项
        self.show_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="显示结果", variable=self.show_var).pack(side=tk.LEFT)
        
        # 功能按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="检测图像", command=self.detect_image).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="摄像头检测", command=self.detect_camera).pack(side=tk.LEFT, padx=(5, 0))
        
        # 结果显示
        result_frame = ttk.LabelFrame(main_frame, text="检测结果")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(result_frame, height=10, width=50)
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def browse_model(self):
        """浏览模型文件"""
        filename = filedialog.askopenfilename(
            title="选择ONNX模型文件",
            filetypes=[("ONNX files", "*.onnx"), ("All files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def load_model(self):
        """加载模型"""
        model_path = self.model_path_var.get()
        if not model_path:
            messagebox.showerror("错误", "请先选择模型文件")
            return
        
        try:
            self.detector = YOLOv8AnimalDetector(
                model_path, 
                conf_threshold=self.conf_var.get(),
                iou_threshold=self.iou_var.get()
            )
            messagebox.showinfo("成功", "模型加载成功！")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {str(e)}")
    
    def detect_image(self):
        """检测图像"""
        if not self.detector:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        filename = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # 更新检测参数
                self.detector.conf_threshold = self.conf_var.get()
                self.detector.iou_threshold = self.iou_var.get()
                
                # 检测动物
                result = self.detector.detect_from_image(filename, show_result=self.show_var.get())
                
                # 显示结果
                self.display_result(f"图像检测结果 ({filename}):", result)
                
            except Exception as e:
                messagebox.showerror("错误", f"检测失败: {str(e)}")
    
    def detect_camera(self):
        """摄像头检测"""
        if not self.detector:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        try:
            # 更新检测参数
            self.detector.conf_threshold = self.conf_var.get()
            self.detector.iou_threshold = self.iou_var.get()
            
            # 开始检测
            result = self.detector.detect_from_camera(
                camera_id=0, 
                show_result=self.show_var.get(), 
                duration=30  # 30秒检测
            )
            
            # 显示结果
            self.display_result("摄像头检测结果 (30秒):", result)
            
        except Exception as e:
            messagebox.showerror("错误", f"摄像头检测失败: {str(e)}")
    
    def display_result(self, title, result):
        """显示检测结果"""
        self.result_text.insert(tk.END, f"\n{title}\n")
        self.result_text.insert(tk.END, f"{json.dumps(result, indent=2, ensure_ascii=False)}\n")
        self.result_text.insert(tk.END, "-" * 50 + "\n")
        self.result_text.see(tk.END)

# 默认模型路径
DEFAULT_MODEL_PATH = "/home/by/ds25/temp/vision/yolo/best9999.onnx"

def start_camera_detection(model_path=DEFAULT_MODEL_PATH, show_result=True, conf_threshold=0.5, iou_threshold=0.5):
    """直接启动摄像头0检测"""
    try:
        print(f"正在加载模型: {model_path}")
        detector = YOLOv8AnimalDetector(model_path, conf_threshold, iou_threshold)
        print("模型加载成功！")
        print("开始摄像头检测，按 'q' 键退出...")
        
        # 开始检测
        result = detector.detect_from_camera(camera_id=0, show_result=show_result)
        
        print(f"\n=== 检测完成 ===")
        print(f"检测到的动物统计:")
        if result:
            for animal, count in result.items():
                print(f"  {animal}: {count}")
        else:
            print("  未检测到任何动物")
        
        return result
        
    except Exception as e:
        print(f"检测失败: {e}")
        print("请检查:")
        print("1. best9999.onnx 模型文件是否存在")
        print("2. 摄像头是否可以正常使用")
        print("3. 是否安装了必要的依赖包")
        return {}

def main():
    """主函数 - 直接启动摄像头检测"""
    # 创建检测器
    detector = YOLOv8AnimalDetector('/home/by/ds25/temp/vision/yolo/best9999.onnx')
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    

    ret, frame = cap.read()
    result = detector.detect_animals(frame, show_result=False)

    def format_animal_counts(animal_dict):
        # 严格定义顺序：elephant(e) → monkey(m) → peacock(p) → wolf(w) → tiger(t)
        order = [
            ('elephant', 'e'),
            ('monkey', 'm'),
            ('peacock', 'p'),
            ('wolf', 'w'),
            ('tiger', 't')
        ]
        # 按每个动物的数量，不存在则为0
        parts = []
        for animal, abbr in order:
            count = animal_dict.get(animal, 0)
            parts.append(f"{abbr}{count}")
        
        # 拼接成最终字符串
        return ''.join(parts)

    
    if not result:
        print("未识别到")
    else:
        print(result)
        print('转换后:')
        print(format_animal_counts(result))
    # 输出示例
    '''
    {
    'elephant': 4,     # 检测到4只大象
    'monkey': 3,       # 检测到3只猴子  
    'peacock': 2,      # 检测到2只孔雀
    'wolf': 1,         # 检测到1只狼
    'tiger': 1         # 检测到1只老虎
    }
        {
    'elephant': 4,     # 检测到4只大象
    'monkey': 3,       # 检测到3只猴子  
    'peacock': 2,      # 检测到2只孔雀
    'wolf': 1,         # 检测到1只狼
    }
    '''

    # cv2.imshow('Detection', frame)
                
    cap.release()
    # cv2.destroyAllWindows()
        
    # if len(sys.argv) > 1 and sys.argv[1] == "--gui":
    #     # 启动GUI版本
    #     root = tk.Tk()
    #     app = AnimalDetectionGUI(root)
    #     root.mainloop()
    # else:
    #     # 直接启动摄像头检测，使用默认模型 best9999.onnx
    #     print("=== YOLOv8 动物检测器 ===")
    #     print(f"使用模型: {DEFAULT_MODEL_PATH}")
        
    #     # 检查是否需要隐藏可视化结果
    #     show_result = False
    #     if len(sys.argv) > 1 and sys.argv[1].lower() in ['false', 'no', '0']:
    #         show_result = False
            
    #     start_camera_detection(show_result=show_result)

if __name__ == "__main__":
    main()