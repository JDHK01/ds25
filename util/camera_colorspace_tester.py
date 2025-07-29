#!/usr/bin/env python3
"""
摄像头颜色空间转换测试工具
支持多种颜色空间转换、通道分析、颜色统计等功能
"""

import cv2
import numpy as np
import argparse
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
import matplotlib.pyplot as plt
from collections import defaultdict

class ColorSpaceTester:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.cap = None
        
        # 当前显示模式
        self.current_mode = 'original'
        self.current_channel = 0
        self.show_histogram = False
        self.show_stats = True
        
        # 窗口名称
        self.main_window = "Color Space Tester"
        self.control_window = "Controls"
        
        # 支持的颜色空间转换
        self.color_spaces = {
            'original': ('Original (BGR)', None, ['B', 'G', 'R']),
            'rgb': ('RGB', cv2.COLOR_BGR2RGB, ['R', 'G', 'B']),
            'hsv': ('HSV', cv2.COLOR_BGR2HSV, ['H', 'S', 'V']),
            'hls': ('HLS', cv2.COLOR_BGR2HLS, ['H', 'L', 'S']),
            'lab': ('LAB', cv2.COLOR_BGR2LAB, ['L', 'A', 'B']),
            'yuv': ('YUV', cv2.COLOR_BGR2YUV, ['Y', 'U', 'V']),
            'ycrcb': ('YCrCb', cv2.COLOR_BGR2YCrCb, ['Y', 'Cr', 'Cb']),
            'xyz': ('XYZ', cv2.COLOR_BGR2XYZ, ['X', 'Y', 'Z']),
            'gray': ('Grayscale', cv2.COLOR_BGR2GRAY, ['Gray']),
        }
        
        # 颜色过滤器
        self.color_filters = {
            'none': 'No Filter',
            'red': 'Red Filter',
            'green': 'Green Filter', 
            'blue': 'Blue Filter',
            'yellow': 'Yellow Filter',
            'cyan': 'Cyan Filter',
            'magenta': 'Magenta Filter'
        }
        
        self.current_filter = 'none'
        
        # 统计信息
        self.color_stats = {}

    def open_camera(self) -> bool:
        """打开摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                print(f"无法打开摄像头 {self.device_id}")
                return False
            
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print(f"摄像头 {self.device_id} 已打开")
            return True
            
        except Exception as e:
            print(f"打开摄像头时发生错误：{e}")
            return False

    def close_camera(self):
        """关闭摄像头"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        plt.close('all')  # 关闭所有matplotlib窗口

    def convert_color_space(self, frame: np.ndarray, space: str) -> np.ndarray:
        """转换颜色空间"""
        if space == 'original':
            return frame
        
        _, conversion, _ = self.color_spaces[space]
        if conversion is None:
            return frame
        
        try:
            if space == 'gray':
                converted = cv2.cvtColor(frame, conversion)
                # 转换为3通道以便显示
                return cv2.cvtColor(converted, cv2.COLOR_GRAY2BGR)
            else:
                return cv2.cvtColor(frame, conversion)
        except Exception as e:
            print(f"颜色空间转换失败 {space}: {e}")
            return frame

    def extract_channel(self, frame: np.ndarray, channel: int) -> np.ndarray:
        """提取单个颜色通道"""
        if len(frame.shape) == 2:  # 灰度图
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        if channel >= frame.shape[2]:
            return frame
        
        # 创建单通道图像
        single_channel = np.zeros_like(frame)
        single_channel[:, :, channel] = frame[:, :, channel]
        
        return single_channel

    def apply_color_filter(self, frame: np.ndarray, filter_type: str) -> np.ndarray:
        """应用颜色过滤器"""
        if filter_type == 'none':
            return frame
        
        # 转换到HSV进行颜色过滤
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 定义颜色范围
        color_ranges = {
            'red': [(np.array([0, 50, 50]), np.array([10, 255, 255])),
                   (np.array([170, 50, 50]), np.array([180, 255, 255]))],
            'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            'blue': [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            'yellow': [(np.array([20, 50, 50]), np.array([40, 255, 255]))],
            'cyan': [(np.array([80, 50, 50]), np.array([100, 255, 255]))],
            'magenta': [(np.array([140, 50, 50]), np.array([170, 255, 255]))]
        }
        
        if filter_type not in color_ranges:
            return frame
        
        # 创建掩码
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges[filter_type]:
            mask += cv2.inRange(hsv, lower, upper)
        
        # 应用掩码
        result = frame.copy()
        result[mask == 0] = [0, 0, 0]  # 非目标颜色变黑
        
        return result

    def calculate_color_stats(self, frame: np.ndarray, space: str) -> Dict:
        """计算颜色统计信息"""
        if len(frame.shape) == 2:
            # 灰度图
            return {
                'mean': [np.mean(frame)],
                'std': [np.std(frame)],
                'min': [np.min(frame)],
                'max': [np.max(frame)]
            }
        
        stats = {}
        for i in range(frame.shape[2]):
            channel_data = frame[:, :, i]
            stats[f'channel_{i}'] = {
                'mean': np.mean(channel_data),
                'std': np.std(channel_data),
                'min': np.min(channel_data),
                'max': np.max(channel_data)
            }
        
        return stats

    def draw_histogram(self, frame: np.ndarray, space: str) -> np.ndarray:
        """绘制直方图"""
        # 创建直方图画布
        hist_height, hist_width = 200, 640
        hist_image = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
        
        if len(frame.shape) == 2:
            # 灰度图直方图
            hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist, 0, hist_height - 20, cv2.NORM_MINMAX)
            
            for i in range(256):
                x = int(i * hist_width / 256)
                y = hist_height - int(hist[i])
                cv2.line(hist_image, (x, hist_height), (x, y), (255, 255, 255), 1)
        else:
            # 彩色图直方图
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR
            channel_names = self.color_spaces[space][2]
            
            for i in range(min(3, frame.shape[2])):
                hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist, 0, hist_height - 40, cv2.NORM_MINMAX)
                
                for j in range(256):
                    x = int(j * hist_width / 256)
                    y = hist_height - 20 - int(hist[j])
                    cv2.line(hist_image, (x, hist_height - 20), (x, y), colors[i], 1)
                
                # 添加通道标签
                cv2.putText(hist_image, channel_names[i], (10 + i * 60, 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
        
        return hist_image

    def draw_color_info(self, frame: np.ndarray, space: str) -> np.ndarray:
        """在画面上绘制颜色信息"""
        info_frame = frame.copy()
        
        # 计算统计信息
        stats = self.calculate_color_stats(frame, space)
        
        # 背景矩形
        overlay = info_frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, info_frame, 0.3, 0, info_frame)
        
        # 显示当前模式
        mode_text = f"Mode: {self.color_spaces[space][0]}"
        cv2.putText(info_frame, mode_text, (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # 显示过滤器
        filter_text = f"Filter: {self.color_filters[self.current_filter]}"
        cv2.putText(info_frame, filter_text, (15, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示统计信息
        if self.show_stats and len(stats) > 0:
            y_pos = 70
            channel_names = self.color_spaces[space][2]
            
            for i, channel_name in enumerate(channel_names):
                if f'channel_{i}' in stats:
                    channel_stats = stats[f'channel_{i}']
                    stats_text = f"{channel_name}: μ={channel_stats['mean']:.1f} σ={channel_stats['std']:.1f}"
                    cv2.putText(info_frame, stats_text, (15, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_pos += 15
        
        return info_frame

    def create_comparison_view(self, original: np.ndarray) -> np.ndarray:
        """创建多颜色空间对比视图"""
        # 选择几个主要的颜色空间进行对比
        spaces_to_show = ['original', 'hsv', 'lab', 'gray']
        
        # 调整每个子窗口的大小
        cell_width, cell_height = 320, 240
        
        # 创建2x2网格
        result = np.zeros((cell_height * 2, cell_width * 2, 3), dtype=np.uint8)
        
        for i, space in enumerate(spaces_to_show):
            row = i // 2
            col = i % 2
            
            # 转换颜色空间
            converted = self.convert_color_space(original, space)
            
            # 调整大小
            resized = cv2.resize(converted, (cell_width, cell_height))
            
            # 添加标签
            space_name = self.color_spaces[space][0]
            cv2.putText(resized, space_name, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 放置到结果图像中
            y1 = row * cell_height
            y2 = y1 + cell_height
            x1 = col * cell_width
            x2 = x1 + cell_width
            
            result[y1:y2, x1:x2] = resized
        
        return result

    def save_colorspace_images(self, frame: np.ndarray, prefix: str = "colorspace"):
        """保存所有颜色空间的图像"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for space_key, (space_name, _, _) in self.color_spaces.items():
            converted = self.convert_color_space(frame, space_key)
            filename = f"{prefix}_{space_key}_{timestamp}.jpg"
            
            try:
                cv2.imwrite(filename, converted)
                print(f"保存 {space_name}: {filename}")
            except Exception as e:
                print(f"保存 {space_name} 失败: {e}")

    def run_interactive_mode(self):
        """运行交互模式"""
        if not self.cap:
            print("摄像头未打开")
            return
        
        print("颜色空间测试工具已启动")
        print("按键控制:")
        print("  1-9: 切换颜色空间")
        print("  0: 多颜色空间对比视图")
        print("  c: 切换通道显示")
        print("  f: 切换颜色过滤器")
        print("  h: 切换直方图显示")
        print("  s: 切换统计信息显示")
        print("  w: 保存所有颜色空间图像")
        print("  i: 打印当前统计信息")
        print("  q: 退出")
        
        space_keys = list(self.color_spaces.keys())
        filter_keys = list(self.color_filters.keys())
        current_space_index = 0
        current_filter_index = 0
        show_comparison = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            if show_comparison:
                # 显示对比视图
                display_frame = self.create_comparison_view(frame)
            else:
                # 当前颜色空间
                current_space = space_keys[current_space_index]
                
                # 转换颜色空间
                converted_frame = self.convert_color_space(frame, current_space)
                
                # 应用颜色过滤器
                filtered_frame = self.apply_color_filter(converted_frame, self.current_filter)
                
                # 提取通道（如果需要）
                if self.current_channel > 0 and self.current_channel <= len(self.color_spaces[current_space][2]):
                    display_frame = self.extract_channel(filtered_frame, self.current_channel - 1)
                else:
                    display_frame = filtered_frame
                
                # 添加信息
                display_frame = self.draw_color_info(display_frame, current_space)
                
                # 显示直方图
                if self.show_histogram:
                    hist_image = self.draw_histogram(filtered_frame, current_space)
                    # 将直方图拼接到主画面下方
                    display_frame = np.vstack([display_frame, hist_image])
            
            cv2.imshow(self.main_window, display_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key >= ord('1') and key <= ord('9'):
                # 切换颜色空间
                index = key - ord('1')
                if index < len(space_keys):
                    current_space_index = index
                    self.current_mode = space_keys[index]
                    show_comparison = False
                    print(f"切换到颜色空间: {self.color_spaces[self.current_mode][0]}")
            elif key == ord('0'):
                # 对比视图
                show_comparison = not show_comparison
                status = "开启" if show_comparison else "关闭"
                print(f"对比视图已{status}")
            elif key == ord('c'):
                # 切换通道
                current_space = space_keys[current_space_index]
                max_channels = len(self.color_spaces[current_space][2])
                self.current_channel = (self.current_channel + 1) % (max_channels + 1)
                
                if self.current_channel == 0:
                    print("显示所有通道")
                else:
                    channel_name = self.color_spaces[current_space][2][self.current_channel - 1]
                    print(f"显示通道: {channel_name}")
            elif key == ord('f'):
                # 切换过滤器
                current_filter_index = (current_filter_index + 1) % len(filter_keys)
                self.current_filter = filter_keys[current_filter_index]
                print(f"切换过滤器: {self.color_filters[self.current_filter]}")
            elif key == ord('h'):
                # 切换直方图
                self.show_histogram = not self.show_histogram
                status = "开启" if self.show_histogram else "关闭"
                print(f"直方图显示已{status}")
            elif key == ord('s'):
                # 切换统计信息
                self.show_stats = not self.show_stats
                status = "开启" if self.show_stats else "关闭"
                print(f"统计信息显示已{status}")
            elif key == ord('w'):
                # 保存图像
                self.save_colorspace_images(frame)
            elif key == ord('i'):
                # 打印统计信息
                current_space = space_keys[current_space_index]
                converted_frame = self.convert_color_space(frame, current_space)
                stats = self.calculate_color_stats(converted_frame, current_space)
                
                print(f"\n=== {self.color_spaces[current_space][0]} 统计信息 ===")
                channel_names = self.color_spaces[current_space][2]
                
                for i, channel_name in enumerate(channel_names):
                    if f'channel_{i}' in stats:
                        channel_stats = stats[f'channel_{i}']
                        print(f"{channel_name} 通道:")
                        print(f"  平均值: {channel_stats['mean']:.2f}")
                        print(f"  标准差: {channel_stats['std']:.2f}")
                        print(f"  最小值: {channel_stats['min']:.2f}")
                        print(f"  最大值: {channel_stats['max']:.2f}")

    def analyze_color_distribution(self, frame: np.ndarray):
        """分析颜色分布"""
        print("\n=== 颜色分布分析 ===")
        
        # 转换到不同颜色空间进行分析
        for space_key, (space_name, _, channel_names) in self.color_spaces.items():
            if space_key == 'original':
                continue
            
            converted = self.convert_color_space(frame, space_key)
            print(f"\n{space_name}:")
            
            for i, channel_name in enumerate(channel_names):
                if len(converted.shape) == 2:  # 灰度图
                    channel_data = converted
                else:
                    if i >= converted.shape[2]:
                        continue
                    channel_data = converted[:, :, i]
                
                # 计算直方图
                hist = cv2.calcHist([channel_data], [0], None, [256], [0, 256])
                
                # 找到峰值
                peak_value = np.argmax(hist)
                peak_count = hist[peak_value][0]
                
                print(f"  {channel_name} 通道: 峰值在 {peak_value} (出现 {int(peak_count)} 次)")


def main():
    parser = argparse.ArgumentParser(description="摄像头颜色空间转换测试工具")
    parser.add_argument('--device', type=int, default=0, help='摄像头设备编号，默认0')
    parser.add_argument('--analyze', action='store_true', help='进行单帧颜色分析')
    parser.add_argument('--save-all', action='store_true', help='保存所有颜色空间图像')
    parser.add_argument('--colorspace', type=str, choices=['original', 'rgb', 'hsv', 'hls', 'lab', 'yuv', 'ycrcb', 'xyz', 'gray'],
                       help='指定颜色空间模式')
    
    args = parser.parse_args()

    tester = ColorSpaceTester(args.device)
    
    try:
        if not tester.open_camera():
            sys.exit(1)
        
        if args.analyze:
            # 单帧分析模式
            print("捕获画面进行颜色分析...")
            ret, frame = tester.cap.read()
            if ret:
                tester.analyze_color_distribution(frame)
                
                if args.save_all:
                    tester.save_colorspace_images(frame)
            else:
                print("无法捕获画面")
        
        elif args.colorspace:
            # 指定颜色空间模式
            tester.current_mode = args.colorspace
            print(f"显示颜色空间: {tester.color_spaces[args.colorspace][0]}")
            
            while True:
                ret, frame = tester.cap.read()
                if not ret:
                    break
                
                converted = tester.convert_color_space(frame, args.colorspace)
                display_frame = tester.draw_color_info(converted, args.colorspace)
                
                cv2.imshow(tester.main_window, display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        else:
            # 交互模式
            tester.run_interactive_mode()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误：{e}")
    finally:
        tester.close_camera()


if __name__ == "__main__":
    main()