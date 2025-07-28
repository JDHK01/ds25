import cv2

def test_camera_crop_manual():
    # 设置裁切参数 (x, y, width, height)
    crop_x = 250   # 从左上角起始 x 坐标
    crop_y = 200   # 从左上角起始 y 坐标
    crop_w = 180   # 宽度
    crop_h = 180   # 高度

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("按 'q' 键退出窗口")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        # 裁切图像
        cropped = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        # 画一个可视框在原图上（可选）
        cv2.rectangle(frame, (crop_x, crop_y), (crop_x + crop_w, crop_y + crop_h), (0, 255, 0), 2)

        # 显示窗口
        cv2.imshow("Camera", frame)
        if cropped.size > 0:
            cv2.imshow("Cropped", cropped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_crop_manual()
