#!/usr/bin/env python3
"""
摄像头工具集GUI启动器
简单的启动脚本，指向util目录中的主程序
"""

import os
import sys
import subprocess

def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    util_dir = os.path.join(current_dir, "util")
    
    # 检查util目录是否存在
    if not os.path.exists(util_dir):
        print("错误: util目录不存在")
        print("请确保摄像头工具位于util/目录中")
        sys.exit(1)
    
    # 检查主GUI文件是否存在
    gui_script = os.path.join(util_dir, "run_camera_tools.py")
    if not os.path.exists(gui_script):
        print("错误: 找不到GUI启动脚本")
        print(f"预期位置: {gui_script}")
        sys.exit(1)
    
    print("启动摄像头工具集GUI...")
    print(f"工具目录: {util_dir}")
    
    try:
        # 切换到util目录并运行GUI
        os.chdir(util_dir)
        os.system(f"{sys.executable} run_camera_tools.py")
    except Exception as e:
        print(f"启动失败: {e}")
        print("\n手动启动方法:")
        print(f"cd {util_dir}")
        print("python run_camera_tools.py")

if __name__ == "__main__":
    main()