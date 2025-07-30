# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a drone control system for competition tasks (边飞行边检测), implementing real-time flight path following with simultaneous target detection and visual navigation. The system combines MAVSDK-Python drone control with computer vision for autonomous drone operations.

## Key Architecture Components

### Core System
- **main.py** - Main flight control loop that orchestrates path following and target detection
- **MAVSDK Integration** - Uses MAVSDK-Python for drone communication and control
- **Vision System** - Computer vision for target detection and visual navigation
- **Path Planning** - Advanced path planning algorithms for optimal flight routes

### Flight Control (`mycontrol/`)
- **control.py** - Basic position control functions with coordinate transformation
- **mission.py** - Mission execution logic

### Path Planning (`gc/`)
- **path_planner.py** - Advanced wildlife patrol system with multiple path planning algorithms:
  - Enhanced greedy algorithm
  - Spiral path planning
  - Zigzag pattern planning
  - Obstacle avoidance and forbidden zone handling
- **ground_station.py** - Ground control station interface
- **drone_communication.py** - Drone communication protocols

### Vision System (`vision/`)
- **mono_camera.py** - Complete vision guidance system with:
  - PID-controlled visual servoing
  - QR code detection for target tracking
  - Two modes: DOWN (vertical alignment) and FRONT (horizontal alignment)
  - Camera offset compensation
  - Real-time task state management
- **yolo/** - YOLO object detection models and inference
- **cv/** - Computer vision utilities

### Utility Tools (`util/`)
- **Camera Tools GUI** - Comprehensive camera testing and configuration suite:
  - Device scanning and performance testing
  - Color space testing and image capture
  - Video recording with multiple formats
  - Real-time parameter adjustment
- **Requirements**: PyQt5, OpenCV, NumPy, psutil, matplotlib

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
# Run path planning visualization
cd gc/
python3 path_planner.py
```

### Vision System Testing
```bash
# Test vision guidance system (standalone)
cd vision/cv/
python3 mono_camera.py

# YOLO model testing
cd vision/yolo/
python3 test.py
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

### Coordinate System Transformations
The system uses a custom coordinate transformation (`mytf()` function in control.py) to convert between drone NED coordinates and the local coordinate system. This is critical for proper navigation.

### Vision System Configuration
The vision guidance system supports two operational modes:
- **TargetMode.DOWN**: Camera pointing downward for vertical alignment
- **TargetMode.FRONT**: Camera pointing forward for horizontal approach

Camera offset compensation is built-in to account for physical camera placement relative to drone center.

### Path Planning Algorithms
The path planner automatically evaluates multiple algorithms and selects the optimal path based on:
- Total path distance
- Coverage completeness  
- Obstacle avoidance
- Revisit minimization

### Testing Strategy
- **test_main.py**: Complete flight simulation without hardware
- **Camera tools**: Comprehensive camera testing and calibration
- **Vision system**: Standalone testing with real-time feedback
- **Path planner**: Visualization and comparison of different algorithms

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