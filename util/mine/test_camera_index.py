import cv2
import argparse

# 创建参数解析器
parser = argparse.ArgumentParser(description="摄像头测试程序")
parser.add_argument('--device', type=int, default=0, help='摄像头设备编号，默认是0')
parser.add_argument('--width', type=int, help='设置图像宽度（例如640）')
parser.add_argument('--height', type=int, help='设置图像高度（例如480）')
parser.add_argument('--crop_x', type=int, help='裁切区域左上角的x坐标')
parser.add_argument('--crop_y', type=int, help='裁切区域左上角的y坐标')
parser.add_argument('--crop_width', type=int, help='裁切区域的宽度')
parser.add_argument('--crop_height', type=int, help='裁切区域的高度')
args = parser.parse_args()

cap = None  # 提前声明，方便在 finally 中使用

try:
    # 打开指定编号的摄像头
    cap = cv2.VideoCapture(args.device)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.device}")

    # 设置图像分辨率（如果指定）
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("按 'q' 键退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("无法读取视频帧")

        # 裁切图像（如果提供了裁切参数）
        if args.crop_x is not None and args.crop_y is not None and args.crop_width is not None and args.crop_height is not None:
            x, y = args.crop_x, args.crop_y
            w, h = args.crop_width, args.crop_height
            # 防止越界
            x2 = min(x + w, frame.shape[1])
            y2 = min(y + h, frame.shape[0])
            frame = frame[y:y2, x:x2]

        # 显示图像
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"发生错误：{e}")

finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
