import Jetson.GPIO as GPIO
import time


# 选择物理引脚编号模式
GPIO.setmode(GPIO.BOARD)

# 设置使用的 GPIO 引脚（物理引脚7）
laser_pin = 7

# 设置为输出
GPIO.setup(laser_pin, GPIO.OUT, initial=GPIO.HIGH)

# 打开激光
GPIO.output(laser_pin, GPIO.HIGH)
print("Laser ON")
time.sleep(6)

# 关闭激光
GPIO.output(laser_pin, GPIO.LOW)
print("Laser OFF")
time.sleep(6)

# 清理资源
GPIO.cleanup()