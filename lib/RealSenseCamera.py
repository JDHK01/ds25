import pyrealsense2 as rs
import numpy as np
import cv2

# 全局变量控制窗口显示
SHOW_WINDOW = False

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        """
        RealSense相机API - 用于其他程序调用
        
        Args:
            width: 图像宽度，默认640
            height: 图像高度，默认480
            fps: 帧率，默认30
        """
        # 配置
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        # 启动管道
        self.pipeline = rs.pipeline()
        self.pipeline.start(config)
        # 获取深度比例
        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # 创建对齐处理器
        self.align = rs.align(rs.stream.color)
        
        # 缓存最新的帧
        self._rgb_frame = None
        self._depth_frame = None
        
        # 窗口相关
        self._window_initialized = False
        
    def _init_windows(self):
        """初始化窗口"""
        if SHOW_WINDOW and not self._window_initialized:
            cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Depth Camera', cv2.WINDOW_AUTOSIZE)
            self._window_initialized = True
    
    def _show_frames(self):
        """显示帧到窗口"""
        if SHOW_WINDOW and self._rgb_frame is not None and self._depth_frame is not None:
            self._init_windows()
            
            # 显示RGB图像
            cv2.imshow('RGB Camera', self._rgb_frame)
            
            # 显示深度图像（彩色映射）
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(self._depth_frame, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.imshow('Depth Camera', depth_colormap)
            
            cv2.waitKey(1)
    
    def update_frames(self):
        """更新帧数据"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if depth_frame and color_frame:
            self._depth_frame = np.asanyarray(depth_frame.get_data())
            self._rgb_frame = np.asanyarray(color_frame.get_data())
            
            # 显示窗口（如果启用）
            self._show_frames()
            
            return True
        return False
    
    def get_rgb_frame(self):
        """
        获取RGB帧
        
        Returns:
            numpy.ndarray: RGB图像数组，BGR格式
        """
        if self.update_frames():
            return self._rgb_frame
        return None
    
    def get_depth_value(self, x, y):
        """
        获取指定坐标的深度值（米）
        等价于 distance = depth_image[y, x] * depth_scale
        
        Args:
            x: 像素x坐标
            y: 像素y坐标
            
        Returns:
            float: 深度值（米），如果获取失败返回None
        """
        if self._depth_frame is None:
            if not self.update_frames():
                return None
                
        if (0 <= x < self._depth_frame.shape[1] and 
            0 <= y < self._depth_frame.shape[0]):
            return self._depth_frame[y, x] * self.depth_scale
        return None
    
    def get_depth_frame(self):
        """
        获取深度帧（原始数据）
        
        Returns:
            numpy.ndarray: 深度图像数组
        """
        if self.update_frames():
            return self._depth_frame
        return None
    
    def close(self):
        """关闭相机"""
        self.pipeline.stop()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数"""
    if event == cv2.EVENT_LBUTTONDOWN:
        camera = param
        distance = camera.get_depth_value(x, y)
        if distance is not None:
            print(f"点击坐标({x}, {y})的深度: {distance:.3f}米")


def main():
    """使用示例"""
    '''
        camera = RealSenseCamera()
        # 获取RGB帧的API
        rgb_frame = camera.get_rgb_frame()
        q# 获取深度的API
        center_depth = camera.get_depth_value(center_x, center_y)
    '''
    global SHOW_WINDOW
    
    # 设置是否显示窗口
    SHOW_WINDOW = True  # 改为False可关闭窗口显示
    # 创建相机实例
    camera = RealSenseCamera()

    try:
            
        print("相机已启动，按 'q' 退出")
        print("点击深度图像可显示该点深度值")
        
        mouse_callback_set = False
        
        while True:
            # 获取RGB帧的API
            rgb_frame = camera.get_rgb_frame()
            if rgb_frame is None:
                continue

            # center_depth = camera.get_depth_value(center_x, center_y)

            if SHOW_WINDOW and not mouse_callback_set and camera._window_initialized:
                cv2.setMouseCallback('Depth Camera', mouse_callback, camera)
                mouse_callback_set = True
            h, w = rgb_frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            # 获取深度的API
            center_depth = camera.get_depth_value(center_x, center_y)
            if center_depth is not None:
                print(f"中心点({center_x}, {center_y})深度: {center_depth:.3f}米", end='\r')
            if SHOW_WINDOW:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                import time
                time.sleep(0.033)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        camera.close()
        print("相机已关闭")


if __name__ == "__main__":
    main()