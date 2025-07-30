#!/usr/bin/env python3
"""
摄像头工具集启动脚本
检查依赖并启动GUI应用程序
"""

import sys
import os
import subprocess
import importlib

def check_dependency(package_name, import_name=None):
    """检查依赖包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_dependency(package_name):
    """安装依赖包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("=" * 60)
    print("摄像头工具集 - Camera Tools Suite")
    print("=" * 60)
    
    # 检查Python版本
    if sys.version_info < (3, 6):
        print("错误: 需要Python 3.6或更高版本")
        print("当前版本:", sys.version_info[:2])
        sys.exit(1)
    
    print(f"Python版本: {sys.version}")
    
    # 检查必需的依赖
    dependencies = [
        ("PyQt5", "PyQt5.QtWidgets"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("psutil", "psutil"),
        ("matplotlib", "matplotlib")
    ]
    
    missing_deps = []
    
    print("\n检查依赖包...")
    for package_name, import_name in dependencies:
        if check_dependency(package_name, import_name):
            print(f"✓ {package_name}")
        else:
            print(f"✗ {package_name} (缺失)")
            missing_deps.append(package_name)
    
    # 如果有缺失的依赖，询问是否安装
    if missing_deps:
        print(f"\n发现缺失的依赖包: {', '.join(missing_deps)}")
        
        try:
            response = input("是否自动安装缺失的依赖? (y/n): ").strip().lower()
            if response in ['y', 'yes', '是']:
                print("\n正在安装依赖包...")
                for package in missing_deps:
                    print(f"安装 {package}...")
                    if install_dependency(package):
                        print(f"✓ {package} 安装成功")
                    else:
                        print(f"✗ {package} 安装失败")
                        print(f"请手动运行: pip install {package}")
                        sys.exit(1)
            else:
                print("请手动安装依赖包后再运行:")
                print(f"pip install {' '.join(missing_deps)}")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\n\n用户取消安装")
            sys.exit(1)
    
    print("\n所有依赖已满足，启动GUI应用程序...")
    
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gui_script = os.path.join(script_dir, "camera_tools_gui.py")
    
    if not os.path.exists(gui_script):
        print(f"错误: 找不到GUI脚本 {gui_script}")
        sys.exit(1)
    
    # 启动GUI应用程序
    try:
        # 导入并运行GUI
        sys.path.insert(0, script_dir)
        from camera_tools_gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"启动GUI时发生错误: {e}")
        print("\n尝试直接运行:")
        print(f"python {gui_script}")
        sys.exit(1)

if __name__ == "__main__":
    main()