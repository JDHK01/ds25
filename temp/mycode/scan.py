import cv2
import time
from typing import List, Optional, Dict
from datetime import datetime

class QRCodeScanner:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.detector = cv2.QRCodeDetector()
        self.is_running = False
        self.last_scan_data = None
        self.last_scan_time = 0
        self.scan_cooldown = 2  # seconds
        self.scan_history = []

    def start(self) -> bool:
        """启动摄像头"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("❌ 无法打开摄像头")
            return False
        self.is_running = True
        print("✅ 摄像头已启动")
        return True

    def stop(self):
        """停止摄像头"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
        print("🛑 摄像头已停止")

    def read_frame(self) -> Optional:
        """读取一帧图像"""
        if not self.is_running or self.cap is None:
            return None
        ret, frame = self.cap.read()
        x, y, w, h = 250, 200, 180, 180
        cropped_frame = frame[y:y+h, x:x+w]
        return cropped_frame if ret else None

    def detect_qrcode_with_info(self, frame) -> Dict:
        """检测二维码并返回信息"""
        data, points, _ = self.detector.detectAndDecode(frame)
        is_number = data.isdigit() if data else False
        return {
            "data": data if data else None,
            "points": points,
            "type": "QR_CODE" if data else None,
            "is_number": is_number,
        }

    def scan_single(self, timeout: float = 10.0) -> Optional[Dict]:
        """扫描一个二维码（阻塞式，等待一定时间）"""
        if not self.is_running:
            print("请先启动摄像头")
            return None

        start_time = time.time()
        print(f"⌛ 等待扫描二维码（{timeout}秒超时）...")

        while (time.time() - start_time) < timeout:
            frame = self.read_frame()
            if frame is None:
                continue

            info = self.detect_qrcode_with_info(frame)
            if info["data"] is not None:
                data_type = "数字" if info["is_number"] else "文本"
                print(f"✅ 成功扫描到{data_type}: {info['data']}")
                return info

        print("⌛ 超时未扫描到二维码")
        return None

    # def scan_continuous(self, callback=None) -> List[Dict]:
    #     """连续扫描二维码，不显示窗口"""
    #     if not self.is_running:
    #         print("请先启动摄像头")
    #         return []

    #     print("\n📡 开始连续扫描... (无窗口显示)")
    #     print("- 按 Ctrl+C 停止")
    #     print("-" * 40)

    #     try:
    #         while True:
    #             frame = self.read_frame()
    #             if frame is None:
    #                 continue

    #             info = self.detect_qrcode_with_info(frame)
    #             current_time = time.time()

    #             if info["data"] is not None:
    #                 if (info["data"] != self.last_scan_data or
    #                     (current_time - self.last_scan_time) > self.scan_cooldown):

    #                     self.last_scan_data = info["data"]
    #                     self.last_scan_time = current_time

    #                     scan_record = {
    #                         "timestamp": datetime.now(),
    #                         "data": info["data"],
    #                         "type": info["type"],
    #                         "is_number": info["is_number"]
    #                     }
    #                     self.scan_history.append(scan_record)

    #                     data_type = "数字" if info["is_number"] else "文本"
    #                     print(f"[{scan_record['timestamp'].strftime('%H:%M:%S')}] 扫描到{data_type}: {info['data']}")

    #                     if callback:
    #                         callback(scan_record)

    #     except KeyboardInterrupt:
    #         print("\n🛑 扫描被用户中断")

    #     return self.scan_history

def quick_scan():
    scanner = QRCodeScanner(camera_index=4)
    if scanner.start():
        print("\n等待扫描数字二维码...")
        result = scanner.scan_single(timeout=10.0)
                
        if result:
            if result['is_number']:
                number = int(result['data'])    
                return number
                print("\n检测到二维码")
        else:
            print("\n未能检测到二维码")
            return -1
        scanner.stop()

if __name__ == "__main__":
    quick_scan()