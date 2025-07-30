#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试地面站和路径规划的集成功能
"""

import sys
sys.path.append('/Users/yqz/by/wrj/gc')

from plan import plan_path_api

def test_path_planning():
    """测试路径规划功能"""
    print("=== 测试路径规划功能 ===")
    
    # 测试案例1：标准3个障碍点
    test_cases = [
        ["A1B2", "A1B3", "A1B4"],
        ["A3B3", "A4B3", "A5B3"],
        ["A2B1", "A3B1", "A4B1"]
    ]
    
    for i, obstacles in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}: {obstacles}")
        try:
            result = plan_path_api(obstacles, "test_output")
            if result["success"]:
                print(f"✓ 成功 - 路径长度: {result['path_length']}步")
                print(f"  障碍点: {result['obstacle_points']}")
                print(f"  起始几步: {' -> '.join(result['path_labels'][:5])}...")
            else:
                print(f"✗ 失败: {result['message']}")
        except Exception as e:
            print(f"✗ 异常: {e}")
    
    print("\n=== 路径规划测试完成 ===")

def test_grid_calculations():
    """测试网格计算"""
    print("=== 测试网格计算 ===")
    
    # 模拟点击坐标转换
    grid_width, grid_height = 450, 350  # 网格控件尺寸
    cell_width = grid_width / 9
    cell_height = grid_height / 7
    
    test_clicks = [
        (10, 10),    # 左上角
        (225, 175),  # 中心
        (440, 340)   # 右下角
    ]
    
    for x, y in test_clicks:
        col = int(x / cell_width)
        row = int(y / cell_height)
        
        if 0 <= col < 9 and 0 <= row < 7:
            grid_code = f"A{col+1}B{row+1}"
            print(f"点击坐标 ({x}, {y}) -> 网格 {grid_code}")
        else:
            print(f"点击坐标 ({x}, {y}) -> 超出范围")
    
    print("=== 网格计算测试完成 ===")

if __name__ == '__main__':
    test_path_planning()
    test_grid_calculations()
    print("\n所有测试已执行完毕！")