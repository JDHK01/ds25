import cv2

# 打开默认摄像头（设备号为0）
cap = cv2.VideoCapture(7)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("按 'q' 键退出")

while True:
    # 逐帧捕获
    ret, frame = cap.read()

    if not ret:
        print("无法读取视频帧")
        break

    # 显示帧内容
    cv2.imshow('Camera', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
