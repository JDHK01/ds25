# 摄像头工具集 GUI 界面

## 🚀 快速开始

已将所有GUI相关文件移动到 `util/` 目录中，与摄像头工具放在一起。

### 启动方法

#### 方法1：使用主目录启动器（推荐）
```bash
# 在主目录运行
python3 launch_camera_gui.py
```

#### 方法2：进入util目录启动
```bash
# 进入工具目录
cd util/

# 运行GUI启动脚本（自动检查依赖）
python3 run_camera_tools.py

# 或直接运行GUI
python3 camera_tools_gui.py
```

## 📁 文件结构

```
wrj/
├── launch_camera_gui.py       # 主目录启动器
└── util/                      # 工具目录
    ├── camera_tools_gui.py    # 主GUI应用程序
    ├── run_camera_tools.py    # 智能启动脚本
    ├── requirements_gui.txt   # GUI依赖包列表
    ├── README_GUI.md          # 详细使用说明
    ├── test_gui_structure.py  # 结构测试脚本
    │
    # 摄像头工具集
    ├── camera_colorspace_tester.py    # 颜色空间测试器
    ├── camera_comparator.py           # 摄像头对比工具
    ├── camera_device_scanner.py       # 设备扫描器
    ├── camera_performance_tester.py   # 性能测试器
    ├── camera_realtime_adjuster.py    # 实时参数调节器
    ├── camera_recorder.py             # 录制工具
    ├── camera_resolution_adjuster.py  # 分辨率调节器
    └── camera_screenshot_tool.py      # 截图工具
```

## 🛠️ 系统要求

- **Python**: 3.6或更高版本
- **依赖包**: PyQt5, OpenCV, NumPy, psutil, matplotlib
- **摄像头**: 至少一个可用的摄像头设备

## 📋 功能列表

### GUI界面功能
- 🔍 **设备扫描** - 自动扫描所有可用摄像头
- 🎨 **颜色空间测试** - 多种颜色空间转换预览
- ⚡ **性能测试** - 全面性能分析和压力测试
- 📹 **视频录制** - 多格式录制和实时预览
- 📸 **截图工具** - 多种截图模式和图像处理

### 独立工具（也可通过GUI访问）
- ⚙️ **实时参数调节器** - GUI滑块调节摄像头参数
- 🔄 **摄像头对比工具** - 多摄像头同时对比
- 📏 **分辨率调节器** - 分辨率测试和调节

## 🔧 安装和配置

### 自动安装（推荐）
启动脚本会自动检查并安装缺失的依赖：
```bash
python3 util/run_camera_tools.py
```

### 手动安装
```bash
# 安装依赖
pip3 install -r util/requirements_gui.txt

# 或单独安装
pip3 install PyQt5 opencv-python numpy psutil matplotlib
```

## 🎯 使用指南

1. **启动应用**
   ```bash
   python3 launch_camera_gui.py
   ```

2. **扫描设备**
   - 在"设备扫描"标签页查看可用摄像头
   - 记住可用的设备ID号

3. **使用各种工具**
   - 选择合适的设备ID
   - 配置相关参数
   - 点击启动按钮开始使用

4. **保存结果**
   - 大部分工具支持保存结果到文件
   - 录制和截图会自动保存到指定目录

## 📸 截图和演示

GUI界面提供了统一的操作体验：
- 标签页设计，功能分类清晰
- 实时摄像头预览
- 参数可视化配置
- 进度显示和状态反馈

## 🐛 故障排除

### 常见问题
1. **无法启动GUI**
   - 检查Python版本（需要3.6+）
   - 安装PyQt5: `pip3 install PyQt5`

2. **摄像头无法打开**
   - 确保摄像头未被其他程序占用
   - 检查设备权限设置

3. **依赖包缺失**
   - 运行: `pip3 install -r util/requirements_gui.txt`
   - 或使用启动脚本自动安装

### 测试和调试
```bash
# 运行结构测试
cd util/
python3 test_gui_structure.py

# 测试单个工具
python3 camera_device_scanner.py --list
```

## 📝 更新日志

### v1.1 (当前版本)
- 文件重新组织到util目录
- 修复路径引用问题
- 添加主目录启动器
- 改进错误处理

### v1.0
- 初始GUI界面实现
- 集成8个摄像头工具
- 提供统一操作界面

## 💡 提示

- 第一次使用建议先运行设备扫描
- 录制大文件时注意磁盘空间
- 性能测试会占用较多系统资源
- 可以同时使用GUI和命令行工具

详细使用说明请查看 `util/README_GUI.md`