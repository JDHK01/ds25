#!/usr/bin/env python3

import sys
sys.path.append('/home/by/wrj/ds24/lib')
import scan, ser
import mycontrol as ctrl
import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)
from mavsdk.telemetry import LandedState

# --- 串口设置 ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600

# --- 发送数据包协议定义 ---
LORA_HEADER_GOODS = "$GOODS,"
LORA_HEADER_X = "$XDATA,"
LORA_HEADER_Y = "$YDATA,"
LORA_HEADER_VELOCITY = "$VELOCITY,"
LORA_HEADER_SEARCH = "$SEARCH,"
LORA_HEADER_SEARCHID = "$$SEARCHID,"
LORA_HEADER_COMMAND_TJC = "%COMMAND,"

# --- 接收数据包协议定义 ---
LORA_HEADER_COMMAND_RECV = "$$COMMAND,"

# --- 通用包尾 ---
LORA_PACKET_FOOTER = ",END#"


async def run():
    drone = System()
    await drone.connect(system_address="udp://127.0.0.1:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for drone to have a local position estimate and be armable...")
    async for health in drone.telemetry.health():
        if health.is_local_position_ok and health.is_armable:
            print("-- Local position estimate OK and drone is armable")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Setting initial setpoint")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 90.0))

    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed \
                with error code: {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return

    print("-- takeoff \
            within local coordinate system")
    
    ser_port = ser.SerialPort(port=SERIAL_PORT, baudrate=BAUD_RATE)
    if not ser_port.open():
        return

    #============================测试===========================
    await ctrl.goto_position_ned(drone, 0.0, 0.0, -1.25, 0.0, 20.0)
    await ctrl.goto_position_ned(drone, 1.4, 0.0, -1.25, 0.0, 10.0)
    await ctrl.goto_position_ned(drone, 1.4, 0.25, -1.25, 0.0, 5.0)

    result = scan.quick_scan()
    ser_port.send_goods_packet(result, 'A1')
    
    await ctrl.goto_position_ned(drone, 1.4, 1.3, -1.25, 0.0, 10.0)
    await ctrl.goto_position_ned(drone, 1.4, 1.3, -0.95, 0.0, 10.0)
    
    result = scan.quick_scan()
    ser_port.send_goods_packet(result, 'A2')
    
    await ctrl.goto_position_ned(drone, 1.4, 1.3, 0, 0.0, 20.0)

    
    

    async for state in drone.telemetry.landed_state():
        if state == LandedState.ON_GROUND:
            break

    # ==============================停止offboard模式========================================
    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except Exception as error:
        print(f"Stopping offboard mode failed with error: {error}")
    # ==============================解锁========================================
    print("-- Disarming")
    await drone.action.disarm()


if __name__ == "__main__":
    ser_port = ser.SerialPort(port=SERIAL_PORT, baudrate=BAUD_RATE)
    if not ser_port.open():
        pass
    result = scan.quick_scan()
    ser_port.send_goods_packet(result, 'A1')
