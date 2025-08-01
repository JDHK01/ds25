import Jetson.GPIO as GPIO
import asyncio

async def led_flash(stop_event):
    """LED 持续闪烁直到 stop_event 被设置"""
    LED_PIN = 32
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    status = GPIO.HIGH

    try:
        while not stop_event.is_set():
            GPIO.output(LED_PIN, status)
            await asyncio.sleep(0.5)
            status = GPIO.LOW if status == GPIO.HIGH else GPIO.HIGH
    finally:
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.cleanup()
        print("[LED] 闪烁停止,GPIO 已清理")
