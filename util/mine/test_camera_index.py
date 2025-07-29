import cv2
import argparse

# 创建参数解析器
parser = argparse.ArgumentParser(description="摄像头测试程序")
parser.add_argument('--device', type=int, default=0, help='摄像头设备编号，默认是0')
args = parser.parse_args()

cap = None  # 提前声明，方便在 finally 中使用

try:
    # 打开指定编号的摄像头
    cap = cv2.VideoCapture(args.device)
    
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.device}")

    print("按 'q' 键退出")

    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("无法读取视频帧")

        # 显示帧内容
        cv2.imshow('Camera', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"发生错误：{e}")

finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
