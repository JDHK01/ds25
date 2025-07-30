#!/usr/bin/env python3
"""
摄像头工具集成GUI界面
使用PyQt5整合util/目录中的所有摄像头工具
"""

import sys
import os
import subprocess
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QTabWidget, QGroupBox, QPushButton, QLabel, 
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QSlider, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QSplitter, QFrame, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QStatusBar, QToolBar, QAction, QMenuBar
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QObject, QSize, QRect
)
from PyQt5.QtGui import (
    QIcon, QPixmap, QFont, QPalette, QColor, QMovie
)

# 尝试导入OpenCV用于摄像头预览
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("警告: OpenCV未安装，部分功能将不可用")

class CameraPreviewWidget(QLabel):
    """摄像头预览widget"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMaximumSize(640, 480)
        self.setScaledContents(True)
        self.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.setText("摄像头预览\n(未连接)")
        self.setAlignment(Qt.AlignCenter)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.device_id = 0
        
    def start_preview(self, device_id: int = 0):
        """开始预览"""
        if not CV2_AVAILABLE:
            self.setText("OpenCV未安装\n无法显示预览")
            return False
            
        try:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(device_id)
            self.device_id = device_id
            
            if self.cap.isOpened():
                self.timer.start(33)  # ~30 FPS
                return True
            else:
                self.setText(f"无法打开摄像头 {device_id}")
                return False
        except Exception as e:
            self.setText(f"预览错误:\n{str(e)}")
            return False
    
    def stop_preview(self):
        """停止预览"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.setText("摄像头预览\n(未连接)")
    
    def update_frame(self):
        """更新帧"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 转换为RGB格式
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                
                # 创建QPixmap
                q_image = QPixmap.fromImage(
                    QPixmap.fromImage(
                        QPixmap.fromImage(rgb_frame.data, w, h, bytes_per_line, QPixmap.Format_RGB888)
                    )
                )
                
                self.setPixmap(q_image)

class DeviceScannerWidget(QWidget):
    """设备扫描器界面"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 控制按钮
        button_layout = QHBoxLayout()
        self.scan_btn = QPushButton("扫描设备")
        self.scan_btn.clicked.connect(self.scan_devices)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.scan_devices)
        
        button_layout.addWidget(self.scan_btn)
        button_layout.addWidget(self.refresh_btn)
        button_layout.addStretch()
        
        # 设备列表表格
        self.device_table = QTableWidget()
        self.device_table.setColumnCount(5)
        self.device_table.setHorizontalHeaderLabels([
            "设备ID", "名称", "状态", "分辨率", "帧率"
        ])
        self.device_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addLayout(button_layout)
        layout.addWidget(self.device_table)
        
        self.setLayout(layout)
        
        # 初始扫描
        self.scan_devices()
    
    def scan_devices(self):
        """扫描设备"""
        self.device_table.setRowCount(0)
        
        if not CV2_AVAILABLE:
            self.device_table.setRowCount(1)
            self.device_table.setItem(0, 0, QTableWidgetItem("N/A"))
            self.device_table.setItem(0, 1, QTableWidgetItem("OpenCV未安装"))
            self.device_table.setItem(0, 2, QTableWidgetItem("不可用"))
            self.device_table.setItem(0, 3, QTableWidgetItem("N/A"))
            self.device_table.setItem(0, 4, QTableWidgetItem("N/A"))
            return
        
        # 扫描0-9号设备
        available_devices = []
        for device_id in range(10):
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    available_devices.append({
                        'id': device_id,
                        'name': f'Camera {device_id}',
                        'status': '可用',
                        'resolution': f'{width}x{height}',
                        'fps': f'{fps:.1f}'
                    })
                cap.release()
        
        # 更新表格
        self.device_table.setRowCount(len(available_devices))
        for row, device in enumerate(available_devices):
            self.device_table.setItem(row, 0, QTableWidgetItem(str(device['id'])))
            self.device_table.setItem(row, 1, QTableWidgetItem(device['name']))
            self.device_table.setItem(row, 2, QTableWidgetItem(device['status']))
            self.device_table.setItem(row, 3, QTableWidgetItem(device['resolution']))
            self.device_table.setItem(row, 4, QTableWidgetItem(device['fps']))

class ColorSpaceTestWidget(QWidget):
    """颜色空间测试界面"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = QGroupBox("控制面板")
        control_layout = QVBoxLayout()
        
        # 设备选择
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("设备ID:"))
        self.device_spinbox = QSpinBox()
        self.device_spinbox.setRange(0, 9)
        device_layout.addWidget(self.device_spinbox)
        control_layout.addLayout(device_layout)
        
        # 颜色空间选择
        colorspace_layout = QHBoxLayout()
        colorspace_layout.addWidget(QLabel("颜色空间:"))
        self.colorspace_combo = QComboBox()
        self.colorspace_combo.addItems([
            "Original (BGR)", "RGB", "HSV", "HLS", "LAB", 
            "YUV", "YCrCb", "XYZ", "Grayscale"
        ])
        colorspace_layout.addWidget(self.colorspace_combo)
        control_layout.addLayout(colorspace_layout)
        
        # 按钮
        self.start_test_btn = QPushButton("启动测试")
        self.start_test_btn.clicked.connect(self.start_colorspace_test)
        self.save_images_btn = QPushButton("保存所有颜色空间图像")
        self.save_images_btn.clicked.connect(self.save_colorspace_images)
        
        control_layout.addWidget(self.start_test_btn)
        control_layout.addWidget(self.save_images_btn)
        control_layout.addStretch()
        
        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(300)
        
        # 右侧预览
        preview_panel = QGroupBox("预览")
        preview_layout = QVBoxLayout()
        
        self.colorspace_preview = CameraPreviewWidget()
        preview_layout.addWidget(self.colorspace_preview)
        
        preview_panel.setLayout(preview_layout)
        
        layout.addWidget(control_panel)
        layout.addWidget(preview_panel)
        
        self.setLayout(layout)
    
    def start_colorspace_test(self):
        """启动颜色空间测试"""
        device_id = self.device_spinbox.value()
        try:
            # 使用subprocess启动颜色空间测试工具
            script_path = os.path.join(os.path.dirname(__file__), "camera_colorspace_tester.py")
            subprocess.Popen([sys.executable, script_path, "--device", str(device_id)])
            
            # 同时启动预览
            self.colorspace_preview.start_preview(device_id)
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"启动颜色空间测试失败: {str(e)}")
    
    def save_colorspace_images(self):
        """保存颜色空间图像"""
        device_id = self.device_spinbox.value()
        try:
            script_path = os.path.join(os.path.dirname(__file__), "camera_colorspace_tester.py")
            subprocess.Popen([sys.executable, script_path, "--device", str(device_id), "--save-all"])
            QMessageBox.information(self, "提示", "正在保存所有颜色空间图像...")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存图像失败: {str(e)}")

class PerformanceTestWidget(QWidget):
    """性能测试界面"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 测试配置
        config_group = QGroupBox("测试配置")
        config_layout = QGridLayout()
        
        # 设备选择
        config_layout.addWidget(QLabel("设备ID:"), 0, 0)
        self.perf_device_spinbox = QSpinBox()
        self.perf_device_spinbox.setRange(0, 9)
        config_layout.addWidget(self.perf_device_spinbox, 0, 1)
        
        # 测试持续时间
        config_layout.addWidget(QLabel("测试时长(秒):"), 1, 0)
        self.duration_spinbox = QSpinBox()
        self.duration_spinbox.setRange(5, 300)
        self.duration_spinbox.setValue(10)
        config_layout.addWidget(self.duration_spinbox, 1, 1)
        
        # 测试类型
        config_layout.addWidget(QLabel("测试类型:"), 2, 0)
        self.test_type_combo = QComboBox()
        self.test_type_combo.addItems(["全面测试", "压力测试", "延迟测试"])
        config_layout.addWidget(self.test_type_combo, 2, 1)
        
        config_group.setLayout(config_layout)
        
        # 按钮
        button_layout = QHBoxLayout()
        self.start_perf_test_btn = QPushButton("开始测试")
        self.start_perf_test_btn.clicked.connect(self.start_performance_test)
        self.save_results_btn = QPushButton("保存结果")
        self.save_results_btn.clicked.connect(self.save_test_results)
        
        button_layout.addWidget(self.start_perf_test_btn)
        button_layout.addWidget(self.save_results_btn)
        button_layout.addStretch()
        
        # 结果显示
        results_group = QGroupBox("测试结果")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        
        layout.addWidget(config_group)
        layout.addLayout(button_layout)
        layout.addWidget(results_group)
        
        self.setLayout(layout)
    
    def start_performance_test(self):
        """开始性能测试"""
        device_id = self.perf_device_spinbox.value()
        duration = self.duration_spinbox.value()
        test_type = self.test_type_combo.currentText()
        
        try:
            script_path = os.path.join(os.path.dirname(__file__), "camera_performance_tester.py")
            
            cmd = [sys.executable, script_path, "--device", str(device_id), "--duration", str(duration)]
            
            if test_type == "压力测试":
                cmd.extend(["--stress", "--stress-duration", str(duration)])
            elif test_type == "延迟测试":
                cmd.append("--latency")
            
            # 启动测试进程
            self.results_text.append(f"开始{test_type}，设备ID: {device_id}，时长: {duration}秒")
            subprocess.Popen(cmd)
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"启动性能测试失败: {str(e)}")
    
    def save_test_results(self):
        """保存测试结果"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存测试结果", f"performance_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text files (*.txt);;All files (*.*)"
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.toPlainText())
                QMessageBox.information(self, "成功", f"结果已保存到: {filename}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存失败: {str(e)}")

class RecorderWidget(QWidget):
    """录制工具界面"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.is_recording = False
        
    def init_ui(self):
        layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = QGroupBox("录制控制")
        control_layout = QVBoxLayout()
        
        # 设备选择
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("设备ID:"))
        self.rec_device_spinbox = QSpinBox()
        self.rec_device_spinbox.setRange(0, 9)
        device_layout.addWidget(self.rec_device_spinbox)
        control_layout.addLayout(device_layout)
        
        # 输出设置
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出文件:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("留空自动生成文件名")
        output_layout.addWidget(self.output_edit)
        self.browse_btn = QPushButton("浏览")
        self.browse_btn.clicked.connect(self.browse_output_file)
        output_layout.addWidget(self.browse_btn)
        control_layout.addLayout(output_layout)
        
        # 格式选择
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("视频格式:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["avi", "mp4", "mov", "mkv", "wmv"])
        format_layout.addWidget(self.format_combo)
        control_layout.addLayout(format_layout)
        
        # 质量设置
        quality_layout = QVBoxLayout()
        quality_layout.addWidget(QLabel("帧率:"))
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(15, 60)
        self.fps_spinbox.setValue(30)
        quality_layout.addWidget(self.fps_spinbox)
        
        quality_layout.addWidget(QLabel("分辨率:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "640x480", "1280x720", "1920x1080", "2560x1440"
        ])
        self.resolution_combo.setCurrentText("1280x720")
        quality_layout.addWidget(self.resolution_combo)
        control_layout.addLayout(quality_layout)
        
        # 录制按钮
        self.record_btn = QPushButton("开始录制")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        
        control_layout.addWidget(self.record_btn)
        
        # 快速录制按钮
        quick_record_layout = QHBoxLayout()
        self.quick_30s_btn = QPushButton("录制30秒")
        self.quick_30s_btn.clicked.connect(lambda: self.quick_record(30))
        self.quick_60s_btn = QPushButton("录制60秒")
        self.quick_60s_btn.clicked.connect(lambda: self.quick_record(60))
        
        quick_record_layout.addWidget(self.quick_30s_btn)
        quick_record_layout.addWidget(self.quick_60s_btn)
        control_layout.addLayout(quick_record_layout)
        
        control_layout.addStretch()
        
        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(350)
        
        # 右侧预览
        preview_panel = QGroupBox("录制预览")
        preview_layout = QVBoxLayout()
        
        self.record_preview = CameraPreviewWidget()
        preview_layout.addWidget(self.record_preview)
        
        # 录制状态
        status_layout = QHBoxLayout()
        self.recording_status = QLabel("未录制")
        self.recording_time = QLabel("00:00:00")
        self.recording_time.setStyleSheet("font-family: monospace; font-size: 14px; font-weight: bold;")
        
        status_layout.addWidget(QLabel("状态:"))
        status_layout.addWidget(self.recording_status)
        status_layout.addStretch()
        status_layout.addWidget(self.recording_time)
        
        preview_layout.addLayout(status_layout)
        preview_panel.setLayout(preview_layout)
        
        layout.addWidget(control_panel)
        layout.addWidget(preview_panel)
        
        self.setLayout(layout)
    
    def browse_output_file(self):
        """浏览输出文件"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "选择输出文件", "", 
            "Video files (*.avi *.mp4 *.mov *.mkv *.wmv);;All files (*.*)"
        )
        if filename:
            self.output_edit.setText(filename)
    
    def toggle_recording(self):
        """切换录制状态"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """开始录制"""
        device_id = self.rec_device_spinbox.value()
        output_file = self.output_edit.text()
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"recording_{timestamp}"
        
        format_ext = self.format_combo.currentText()
        fps = self.fps_spinbox.value()
        resolution = self.resolution_combo.currentText()
        width, height = map(int, resolution.split('x'))
        
        try:
            script_path = os.path.join(os.path.dirname(__file__), "camera_recorder.py")
            cmd = [
                sys.executable, script_path,
                "--device", str(device_id),
                "--output", output_file,
                "--format", format_ext,
                "--fps", str(fps),
                "--width", str(width),
                "--height", str(height)
            ]
            
            subprocess.Popen(cmd)
            
            self.is_recording = True
            self.record_btn.setText("停止录制")
            self.record_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
            self.recording_status.setText("录制中...")
            self.recording_status.setStyleSheet("color: red; font-weight: bold;")
            
            # 启动预览
            self.record_preview.start_preview(device_id)
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"开始录制失败: {str(e)}")
    
    def stop_recording(self):
        """停止录制"""
        self.is_recording = False
        self.record_btn.setText("开始录制")
        self.record_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.recording_status.setText("已停止")
        self.recording_status.setStyleSheet("color: black;")
        
        # 停止预览
        self.record_preview.stop_preview()
    
    def quick_record(self, duration: int):
        """快速录制"""
        device_id = self.rec_device_spinbox.value()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"quick_record_{duration}s_{timestamp}"
        format_ext = self.format_combo.currentText()
        
        try:
            script_path = os.path.join(os.path.dirname(__file__), "camera_recorder.py")
            cmd = [
                sys.executable, script_path,
                "--device", str(device_id),
                "--output", output_file,
                "--format", format_ext,
                "--duration", str(duration)
            ]
            
            subprocess.Popen(cmd)
            QMessageBox.information(self, "提示", f"开始录制{duration}秒视频...")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"快速录制失败: {str(e)}")

class ScreenshotWidget(QWidget):
    """截图工具界面"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = QGroupBox("截图控制")
        control_layout = QVBoxLayout()
        
        # 设备选择
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("设备ID:"))
        self.screen_device_spinbox = QSpinBox()
        self.screen_device_spinbox.setRange(0, 9)
        device_layout.addWidget(self.screen_device_spinbox)
        control_layout.addLayout(device_layout)
        
        # 输出目录
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出目录:"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText("screenshots")
        output_layout.addWidget(self.output_dir_edit)
        self.browse_dir_btn = QPushButton("浏览")
        self.browse_dir_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(self.browse_dir_btn)
        control_layout.addLayout(output_layout)
        
        # 图像设置
        image_settings_group = QGroupBox("图像设置")
        image_layout = QGridLayout()
        
        image_layout.addWidget(QLabel("格式:"), 0, 0)
        self.image_format_combo = QComboBox()
        self.image_format_combo.addItems(["jpg", "png", "bmp", "tiff"])
        image_layout.addWidget(self.image_format_combo, 0, 1)
        
        image_layout.addWidget(QLabel("质量:"), 1, 0)
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(10, 100)
        self.quality_slider.setValue(95)
        self.quality_label = QLabel("95")
        self.quality_slider.valueChanged.connect(lambda v: self.quality_label.setText(str(v)))
        image_layout.addWidget(self.quality_slider, 1, 1)
        image_layout.addWidget(self.quality_label, 1, 2)
        
        image_layout.addWidget(QLabel("滤镜:"), 2, 0)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["none", "blur", "sharpen", "emboss", "edge", "sepia", "vintage"])
        image_layout.addWidget(self.filter_combo, 2, 1)
        
        image_settings_group.setLayout(image_layout)
        control_layout.addWidget(image_settings_group)
        
        # 截图按钮
        screenshot_buttons_layout = QVBoxLayout()
        
        self.single_shot_btn = QPushButton("单张截图")
        self.single_shot_btn.clicked.connect(self.take_single_screenshot)
        screenshot_buttons_layout.addWidget(self.single_shot_btn)
        
        self.burst_shot_btn = QPushButton("连续截图(5张)")
        self.burst_shot_btn.clicked.connect(self.take_burst_screenshots)
        screenshot_buttons_layout.addWidget(self.burst_shot_btn)
        
        self.timed_shot_btn = QPushButton("定时截图")
        self.timed_shot_btn.clicked.connect(self.start_timed_screenshots)
        screenshot_buttons_layout.addWidget(self.timed_shot_btn)
        
        control_layout.addLayout(screenshot_buttons_layout)
        control_layout.addStretch()
        
        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(350)
        
        # 右侧预览
        preview_panel = QGroupBox("截图预览")
        preview_layout = QVBoxLayout()
        
        self.screenshot_preview = CameraPreviewWidget()
        preview_layout.addWidget(self.screenshot_preview)
        
        # 启动预览按钮
        preview_btn_layout = QHBoxLayout()
        self.start_preview_btn = QPushButton("启动预览")
        self.start_preview_btn.clicked.connect(self.start_preview)
        self.stop_preview_btn = QPushButton("停止预览")
        self.stop_preview_btn.clicked.connect(self.stop_preview)
        
        preview_btn_layout.addWidget(self.start_preview_btn)
        preview_btn_layout.addWidget(self.stop_preview_btn)
        preview_layout.addLayout(preview_btn_layout)
        
        preview_panel.setLayout(preview_layout)
        
        layout.addWidget(control_panel)
        layout.addWidget(preview_panel)
        
        self.setLayout(layout)
    
    def browse_output_dir(self):
        """浏览输出目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self.output_dir_edit.setText(directory)
    
    def start_preview(self):
        """启动预览"""
        device_id = self.screen_device_spinbox.value()
        self.screenshot_preview.start_preview(device_id)
    
    def stop_preview(self):
        """停止预览"""
        self.screenshot_preview.stop_preview()
    
    def take_single_screenshot(self):
        """单张截图"""
        self._launch_screenshot_tool()
    
    def take_burst_screenshots(self):
        """连续截图"""
        self._launch_screenshot_tool(["--burst", "5"])
    
    def start_timed_screenshots(self):
        """定时截图"""
        self._launch_screenshot_tool(["--interval", "5"])
    
    def _launch_screenshot_tool(self, extra_args=None):
        """启动截图工具"""
        device_id = self.screen_device_spinbox.value()
        output_dir = self.output_dir_edit.text()
        image_format = self.image_format_combo.currentText()
        quality = self.quality_slider.value()
        filter_type = self.filter_combo.currentText()
        
        try:
            script_path = os.path.join(os.path.dirname(__file__), "camera_screenshot_tool.py")
            cmd = [
                sys.executable, script_path,
                "--device", str(device_id),
                "--output-dir", output_dir,
                "--format", image_format,
                "--quality", str(quality),
                "--filter", filter_type
            ]
            
            if extra_args:
                cmd.extend(extra_args)
            
            subprocess.Popen(cmd)
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"启动截图工具失败: {str(e)}")

class CameraToolsMainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("摄像头工具集 - Camera Tools Suite")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建工具栏
        self.create_toolbar()
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        
        # 创建中心widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 添加各种工具标签页
        self.tab_widget.addTab(DeviceScannerWidget(), "设备扫描")
        self.tab_widget.addTab(ColorSpaceTestWidget(), "颜色空间测试")
        self.tab_widget.addTab(PerformanceTestWidget(), "性能测试")
        self.tab_widget.addTab(RecorderWidget(), "视频录制")
        self.tab_widget.addTab(ScreenshotWidget(), "截图工具")
        
        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.tab_widget)
        central_widget.setLayout(layout)
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #2196F3;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu('工具')
        
        camera_adjuster_action = QAction('实时参数调节器', self)
        camera_adjuster_action.triggered.connect(self.launch_camera_adjuster)
        tools_menu.addAction(camera_adjuster_action)
        
        camera_comparator_action = QAction('摄像头对比工具', self)
        camera_comparator_action.triggered.connect(self.launch_camera_comparator)
        tools_menu.addAction(camera_comparator_action)
        
        resolution_adjuster_action = QAction('分辨率调节器', self)
        resolution_adjuster_action.triggered.connect(self.launch_resolution_adjuster)
        tools_menu.addAction(resolution_adjuster_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助')
        
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # 设备扫描按钮
        scan_action = QAction('扫描设备', self)
        scan_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(0))
        toolbar.addAction(scan_action)
        
        toolbar.addSeparator()
        
        # 快速启动按钮
        quick_record_action = QAction('快速录制', self)
        quick_record_action.triggered.connect(self.quick_record)
        toolbar.addAction(quick_record_action)
        
        quick_screenshot_action = QAction('快速截图', self)
        quick_screenshot_action.triggered.connect(self.quick_screenshot)
        toolbar.addAction(quick_screenshot_action)
    
    def launch_camera_adjuster(self):
        """启动摄像头参数调节器"""
        try:
            script_path = os.path.join(os.path.dirname(__file__), "camera_realtime_adjuster.py")
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            QMessageBox.warning(self, "错误", f"启动参数调节器失败: {str(e)}")
    
    def launch_camera_comparator(self):
        """启动摄像头对比工具"""
        try:
            script_path = os.path.join(os.path.dirname(__file__), "camera_comparator.py")
            subprocess.Popen([sys.executable, script_path, "0", "1"])  # 默认对比0号和1号设备
        except Exception as e:
            QMessageBox.warning(self, "错误", f"启动对比工具失败: {str(e)}")
    
    def launch_resolution_adjuster(self):
        """启动分辨率调节器"""
        try:
            script_path = os.path.join(os.path.dirname(__file__), "camera_resolution_adjuster.py")
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            QMessageBox.warning(self, "错误", f"启动分辨率调节器失败: {str(e)}")
    
    def quick_record(self):
        """快速录制"""
        self.tab_widget.setCurrentIndex(3)  # 切换到录制标签页
        self.status_bar.showMessage("切换到录制工具")
    
    def quick_screenshot(self):
        """快速截图"""
        self.tab_widget.setCurrentIndex(4)  # 切换到截图标签页
        self.status_bar.showMessage("切换到截图工具")
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于", 
                         "摄像头工具集 v1.0\n\n"
                         "整合了多种摄像头相关工具:\n"
                         "• 设备扫描和管理\n"
                         "• 颜色空间转换测试\n"
                         "• 性能测试和分析\n"
                         "• 视频录制\n"
                         "• 截图和图像处理\n"
                         "• 实时参数调节\n\n"
                         "基于PyQt5开发")
    
    def closeEvent(self, event):
        """关闭事件"""
        reply = QMessageBox.question(self, '确认', '确定要退出吗？',
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 停止所有预览
            for i in range(self.tab_widget.count()):
                widget = self.tab_widget.widget(i)
                if hasattr(widget, 'colorspace_preview'):
                    widget.colorspace_preview.stop_preview()
                if hasattr(widget, 'record_preview'):
                    widget.record_preview.stop_preview()
                if hasattr(widget, 'screenshot_preview'):
                    widget.screenshot_preview.stop_preview()
            
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("Camera Tools Suite")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Camera Tools")
    
    # 创建主窗口
    window = CameraToolsMainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()