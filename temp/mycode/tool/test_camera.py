#!/usr/bin/env python3
"""
摄像头检测工具 - Ubuntu专用版本
检测可用摄像头并获取详细参数信息
"""

import cv2
import os
import subprocess
import sys
import glob
from typing import List, Dict, Optional, Tuple


class CameraDetector:
    def __init__(self):
        self.working_cameras = []
        self.camera_info = {}
    
    def detect_v4l2_devices(self) -> List[str]:
        """检测Linux v4l2视频设备"""
        video_devices = []
        try:
            # 查找/dev/video*设备
            devices = glob.glob('/dev/video*')
            for device in sorted(devices):
                if os.path.exists(device):
                    video_devices.append(device)
        except Exception as e:
            print(f"检测v4l2设备时出错: {e}")
        
        return video_devices
    
    def get_v4l2_info(self, device_path: str) -> Dict:
        """获取v4l2设备详细信息"""
        info = {}
        try:
            # 使用v4l2-ctl获取设备信息
            cmd = f"v4l2-ctl -d {device_path} --info"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                info['device_info'] = result.stdout.strip()
            
            # 获取支持的格式
            cmd = f"v4l2-ctl -d {device_path} --list-formats-ext"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                info['formats'] = result.stdout.strip()
                
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    def get_camera_properties(self, cap: cv2.VideoCapture) -> Dict:
        """获取摄像头详细属性"""
        properties = {}
        
        # 基本属性列表
        prop_list = [
            (cv2.CAP_PROP_FRAME_WIDTH, 'width'),
            (cv2.CAP_PROP_FRAME_HEIGHT, 'height'), 
            (cv2.CAP_PROP_FPS, 'fps'),
            (cv2.CAP_PROP_FOURCC, 'fourcc'),
            (cv2.CAP_PROP_BRIGHTNESS, 'brightness'),
            (cv2.CAP_PROP_CONTRAST, 'contrast'),
            (cv2.CAP_PROP_SATURATION, 'saturation'),
            (cv2.CAP_PROP_HUE, 'hue'),
            (cv2.CAP_PROP_GAIN, 'gain'),
            (cv2.CAP_PROP_EXPOSURE, 'exposure'),
            (cv2.CAP_PROP_AUTO_EXPOSURE, 'auto_exposure'),
        ]
        
        for prop_id, prop_name in prop_list:
            try:
                value = cap.get(prop_id)
                if prop_name == 'fourcc':
                    # 将FOURCC转换为可读格式
                    fourcc_int = int(value)
                    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
                    properties[prop_name] = f"{fourcc_str} ({fourcc_int})"
                else:
                    properties[prop_name] = value
            except Exception:
                properties[prop_name] = "不支持"
                
        return properties
    
    def test_resolutions(self, cap: cv2.VideoCapture) -> List[Tuple[int, int]]:
        """测试支持的分辨率"""
        common_resolutions = [
            (640, 480),     # VGA
            (800, 600),     # SVGA
            (1024, 768),    # XGA
            (1280, 720),    # HD
            (1920, 1080),   # Full HD
            (2560, 1440),   # QHD
            (3840, 2160),   # 4K
        ]
        
        supported_resolutions = []
        original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        for width, height in common_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if actual_width == width and actual_height == height:
                ret, frame = cap.read()
                if ret and frame is not None:
                    supported_resolutions.append((width, height))
        
        # 恢复原始分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)
        
        return supported_resolutions
    
    def detect_opencv_cameras(self, max_cameras: int = 10) -> None:
        """使用OpenCV检测摄像头"""
        print("🔍 使用OpenCV检测摄像头...")
        
        for index in range(max_cameras):
            print(f"  测试摄像头索引 {index}...", end="")
            
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                print(" ❌")
                continue
            
            # 尝试读取一帧
            ret, frame = cap.read()
            if not ret or frame is None:
                print(" ⚠️ (打开但无法读取)")
                cap.release()
                continue
            
            print(" ✅")
            self.working_cameras.append(index)
            
            # 获取详细信息
            properties = self.get_camera_properties(cap)
            supported_resolutions = self.test_resolutions(cap)
            
            self.camera_info[index] = {
                'properties': properties,
                'supported_resolutions': supported_resolutions,
                'frame_shape': frame.shape,
                'backend': cap.getBackendName() if hasattr(cap, 'getBackendName') else 'Unknown'
            }
            
            cap.release()
    
    def print_detailed_info(self) -> None:
        """打印详细的摄像头信息"""
        if not self.working_cameras:
            print("\n❌ 未检测到可用摄像头")
            return
        
        print(f"\n📷 检测到 {len(self.working_cameras)} 个可用摄像头:")
        print("=" * 60)
        
        for cam_id in self.working_cameras:
            info = self.camera_info[cam_id]
            print(f"\n📹 摄像头 {cam_id}:")
            print(f"  后端: {info['backend']}")
            print(f"  帧尺寸: {info['frame_shape']}")
            
            print("  📊 属性:")
            for prop, value in info['properties'].items():
                if isinstance(value, float):
                    print(f"    {prop}: {value:.2f}")
                else:
                    print(f"    {prop}: {value}")
            
            print("  📐 支持的分辨率:")
            if info['supported_resolutions']:
                for width, height in info['supported_resolutions']:
                    print(f"    {width}x{height}")
            else:
                print("    无标准分辨率支持")
    
    def check_dependencies(self) -> bool:
        """检查系统依赖"""
        print("🔧 检查系统依赖...")
        
        # 检查v4l2-ctl
        try:
            subprocess.run(['v4l2-ctl', '--version'], 
                         capture_output=True, check=True)
            print("  ✅ v4l2-utils 已安装")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  ⚠️ v4l2-utils 未安装")
            print("     运行: sudo apt install v4l-utils")
            return False
    
    def run_full_detection(self) -> None:
        """运行完整的摄像头检测"""
        print("🎥 摄像头检测工具 - Ubuntu版")
        print("=" * 40)
        
        # 检查依赖
        has_v4l2 = self.check_dependencies()
        
        # 检测v4l2设备
        if has_v4l2:
            print("\n🔍 检测v4l2设备...")
            v4l2_devices = self.detect_v4l2_devices()
            if v4l2_devices:
                print(f"  发现 {len(v4l2_devices)} 个视频设备:")
                for device in v4l2_devices:
                    print(f"    {device}")
            else:
                print("  未发现v4l2设备")
        
        # 使用OpenCV检测
        self.detect_opencv_cameras()
        
        # 打印详细信息
        self.print_detailed_info()
        
        # 提供使用建议
        if self.working_cameras:
            print(f"\n💡 使用建议:")
            print(f"  推荐使用摄像头索引: {self.working_cameras[0]}")
            print(f"  示例代码:")
            print(f"    cap = cv2.VideoCapture({self.working_cameras[0]})")
    
    def test_camera_live(self, camera_id: int, duration: int = 5) -> None:
        """实时测试指定摄像头"""
        if camera_id not in self.working_cameras:
            print(f"❌ 摄像头 {camera_id} 不可用")
            return
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头 {camera_id}")
            return
        
        print(f"📹 测试摄像头 {camera_id} ({duration}秒)")
        print("按 'q' 退出，按 's' 截图")
        
        frame_count = 0
        import time
        start_time = time.time()
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 添加信息覆盖
            cv2.putText(frame, f"Camera {camera_id} - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(f"Camera {camera_id} Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"camera_{camera_id}_screenshot.jpg"
                cv2.imwrite(filename, frame)
                print(f"截图已保存: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print(f"实际FPS: {fps:.2f}")


def main():
    detector = CameraDetector()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test" and len(sys.argv) > 2:
            # 测试特定摄像头
            try:
                camera_id = int(sys.argv[2])
                detector.detect_opencv_cameras()
                detector.test_camera_live(camera_id)
            except ValueError:
                print("❌ 请提供有效的摄像头ID")
        else:
            print("用法:")
            print("  python camera_detector.py           # 完整检测")
            print("  python camera_detector.py --test N  # 测试摄像头N")
    else:
        # 运行完整检测
        detector.run_full_detection()


if __name__ == "__main__":
    main()