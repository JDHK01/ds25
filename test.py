import cv2
import time

# 打开摄像头（设备索引 0）
cap = cv2.VideoCapture(0)

# 设置期望的分辨率（可选）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 检查实际设置结果
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"实际分辨率：{int(width)}x{int(height)}")

# 测试帧率：读取 N 帧所需时间
frame_count = 10
start_time = time.time()

for _ in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        print("摄像头读取失败")
        break

end_time = time.time()
elapsed = end_time - start_time

fps = frame_count / elapsed
print(f"平均帧率：{fps:.2f} FPS")

cap.release()
