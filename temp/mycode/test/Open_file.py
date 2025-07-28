import subprocess

def run_script():
    """Run a shell script to start the LiDAR system."""
    # 定义脚本路径
    script_path = '/home/by/livox_ws/lidar_start.sh'

    # 使用 subprocess.run 执行脚本
    try:
        subprocess.run(['bash', script_path], check=True)
        print(f"成功执行脚本: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"执行脚本时出错: {e}")
    except FileNotFoundError:
        print(f"找不到脚本文件: {script_path}")
if __name__ == "__main__":
    run_script()