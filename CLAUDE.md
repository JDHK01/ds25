# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a competition-grade drone control system for "边飞行边检测" (flight while detecting) operations. The system orchestrates autonomous flight path following with simultaneous computer vision-based target detection and visual navigation, designed specifically for wildlife monitoring competitions.

## Key Architecture Components

### Core System
- **main.py/main_example.py** - Primary flight control orchestrators that integrate all subsystems
- **drone_ctrl.py** - Alternative drone controller with A1B1-A9B7 grid waypoint system
- **MAVSDK Integration** - Uses MAVSDK-Python for MAVLink communication and offboard control
- **Serial Communication** - Ground station command integration via `lib/ser.py`

### Flight Control (`mycontrol/`)
- **control.py** - Core position/velocity control with critical `mytf()` coordinate transformation function
- **flightpath.py** - Sophisticated flight path management with waypoint status tracking (PENDING → IN_PROGRESS → ARRIVED → COMPLETED)
- **mission.py** - Mission execution logic and progress tracking
- **drone_ctrl.py** - Path label system with A1B1-A9B7 grid mapping to NED coordinates

### Path Planning (`gc/`)
- **plan_pro_max.py** - Advanced wildlife patrol system with multiple optimization algorithms:
  - Enhanced greedy algorithm
  - Spiral path planning
  - Zigzag pattern planning
  - Obstacle avoidance and forbidden zone handling
- **visual.py** - Path visualization and comparison tools
- **output/** - Generated path files and visualizations

### Vision System (`vision/`)
- **cv/mono_camera.py** - Complete vision guidance system with PID-controlled visual servoing
- **yolo/detect.py** - YOLO inference engine with ONNX runtime for animal detection
- **yolo/best9999.onnx** - Trained model for competition animals: ['elephant', 'monkey', 'peacock', 'wolf', 'tiger']
- **detect_manager.py** - Unified detection interface integrating YOLO with flight system

### Hardware Integration (`lib/`)
- **ser.py** - Serial communication for real-time waypoint commands from ground station
- **current_position.py** - Position tracking and coordinate system management
- **RealSenseCamera.py** - Intel RealSense camera integration

### Utility Tools (`util/`)
- **Comprehensive camera tools suite** with PyQt5 GUIs for device scanning, performance testing, and video recording

## Development Commands

### Running the Main System
```bash
# Main drone control system
python3 main.py

# Test flight simulation (no hardware required)
python3 test_main.py
```

### Path Planning
```bash
# Run advanced path planning with visualization and algorithm comparison
cd gc/
python3 plan_pro_max.py
```

### Vision System Testing
```bash
# Test vision guidance system (standalone)
cd vision/cv/
python3 mono_camera.py

# YOLO animal detection testing
cd vision/yolo/
python3 detect.py
```

### Camera Tools
```bash
# Launch camera tools GUI
cd util/
python3 run_camera_tools.py

# Or use the main launcher
python3 launch_camera_gui.py

# Individual camera tools
python3 camera_device_scanner.py
python3 camera_performance_tester.py
```

### Dependencies Installation
```bash
# GUI tools dependencies
pip3 install -r util/requirements_gui.txt

# Camera-related dependencies  
pip3 install -r temp/mycode/tool/requirements_camera.txt

# Core dependencies (MAVSDK, OpenCV, etc.)
pip3 install mavsdk opencv-python numpy matplotlib psutil PyQt5
```

## Important Development Notes

### Critical Bugs to Avoid
- **mytf() function**: Must use `ctrl.Drone_Controller.mytf()` or import properly - function converts NED coordinates to local coordinate system
- **pilot_plan() method**: Takes two parameters (drone, ser_port) but internally initializes camera and detector
- **Array indexing**: In drone_ctrl.py:275, `self.path_label[i]` can cause IndexError - use `self.path_label[i-1]` instead

### Coordinate System Architecture
- **NED to Local**: Uses `mytf()` function in control.py for coordinate transformation between drone NED and local grid system
- **A1B1-A9B7 Grid**: Waypoint system maps to precise NED coordinates via `label_map` dictionary
- **Competition Format**: Serial communication sends animal detection results in specific format (e.g., "A8B1e2m0p1w0t0")

### Flight Control Flow
1. **Serial Command Reception** → **Path Planning** → **Waypoint Navigation** → **Target Detection** → **Visual Servoing** → **Result Transmission**
2. **One-time Detection**: Each waypoint processed exactly once via `visit_status` tracking
3. **Emergency Landing**: Automatic landing sequences from specific waypoints (A8B1, A9B2)

### Vision System Integration
- **Detection Pipeline**: YOLO animal detection → Visual servoing approach → Continue mission
- **Competition Animals**: Specifically trained for ['elephant', 'monkey', 'peacock', 'wolf', 'tiger']
- **Real-time Processing**: 50Hz control loop with camera buffer optimization

### Hardware Dependencies
- **Serial Port**: `/dev/ttyUSB0` at 9600 baud for ground station communication
- **Camera Device**: Default ID 0 with 640x480 resolution and 1-frame buffer
- **MAVSDK Connection**: `udp://127.0.0.1:14540` for simulation, configurable for hardware

## Hardware Integration

### Drone Communication
- Uses MAVSDK-Python for MAVLink communication
- Default connection: `udp://127.0.0.1:14540` (simulator)
- Supports real hardware through different connection strings

### Camera Setup
- Default camera device ID: 0
- Configurable resolution, FPS, and buffer settings
- Built-in device scanning and performance testing tools

### Real-time Control Loop
- 50Hz control frequency for smooth operation
- Async/await pattern for concurrent operations
- Timeout mechanisms for safety and reliability