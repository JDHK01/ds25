#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标列表分析和可视化脚本
遍历plan_manual.py中的coordinate_list，使用plan_pro_max.py分析，用visual.py可视化
"""

import sys
import os
import matplotlib.pyplot as plt

# 添加gc目录到路径
sys.path.append('/Users/yqz/by/wrj/gc')

# 导入所需模块
from plan_manual import coordinate_list
from plan_pro_max import get_mapping_result
from visual import visualize_grid_navigation

def analyze_and_visualize_coordinates():
    """
    分析coordinate_list中的每个坐标组合并生成可视化图像
    """
    print(f"开始分析 {len(coordinate_list)} 个坐标组合...")
    
    # 创建输出目录
    output_dir = "/Users/yqz/by/wrj/picture"
    os.makedirs(output_dir, exist_ok=True)
    
    successful_analyses = 0
    failed_analyses = 0
    
    # 遍历coordinate_list中的每个坐标组合
    for i, coords in enumerate(coordinate_list):
        print(f"\n处理第 {i+1}/{len(coordinate_list)} 个坐标组合: {coords}")
        
        try:
            # 使用plan_pro_max.py分析坐标
            route_path = get_mapping_result(coords)
            
            if route_path is None:
                print(f"  警告: 无法找到坐标组合 {coords} 的映射结果")
                failed_analyses += 1
                continue
            
            print(f"  找到路径，包含 {len(route_path)} 个点")
            
            # 使用visual.py进行可视化
            # 将当前分析的坐标组合作为灰色区域（障碍物）
            gray_cells = coords.copy()
            
            # 创建可视化
            fig, ax = visualize_grid_navigation(gray_cells, route_path)
            
            # 添加标题信息
            coord_str = "_".join(coords)
            plt.title(f'路径规划可视化 - 第{i+1}组\n障碍区域: {coords}\n路径长度: {len(route_path)}点', 
                     fontsize=12, fontweight='bold')
            
            # 保存图像
            filename = f"coordinate_analysis_{i+1:03d}_{coord_str}.png"
            filepath = os.path.join(output_dir, filename)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()  # 关闭图形以释放内存
            
            print(f"  保存图像: {filename}")
            successful_analyses += 1
            
        except Exception as e:
            print(f"  错误: 处理坐标组合 {coords} 时发生异常: {str(e)}")
            failed_analyses += 1
            continue
    
    # 生成汇总报告
    print(f"\n=== 分析完成 ===")
    print(f"总计处理: {len(coordinate_list)} 个坐标组合")
    print(f"成功分析: {successful_analyses} 个")
    print(f"失败分析: {failed_analyses} 个")
    print(f"图像保存路径: {output_dir}")
    
    # 生成一个汇总图像，显示所有分析的统计信息
    create_summary_visualization(coordinate_list, successful_analyses, failed_analyses, output_dir)

def create_summary_visualization(coordinate_list, successful, failed, output_dir):
    """
    创建汇总可视化图像
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：成功/失败统计
    labels = ['成功分析', '失败分析']
    sizes = [successful, failed]
    colors = ['green', 'red'] if failed > 0 else ['green']
    
    if failed > 0:
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    else:
        ax1.pie([successful], labels=['成功分析'], colors=['green'], autopct='%1.1f%%', startangle=90)
    ax1.set_title('分析结果统计', fontsize=14, fontweight='bold')
    
    # 右图：坐标分布分析
    # 统计A和B的坐标分布
    a_coords = []
    b_coords = []
    
    for coords in coordinate_list:
        for coord in coords:
            # 解析坐标格式 'AnBm'
            a_pos = coord.find('A')
            b_pos = coord.find('B')
            if a_pos != -1 and b_pos != -1:
                a_val = int(coord[a_pos+1:b_pos])
                b_val = int(coord[b_pos+1:])
                a_coords.append(a_val)
                b_coords.append(b_val)
    
    # 绘制分布直方图
    ax2.hist(a_coords, bins=range(1, 11), alpha=0.7, label='A坐标分布', color='blue')
    ax2.hist(b_coords, bins=range(1, 9), alpha=0.7, label='B坐标分布', color='orange')
    ax2.set_xlabel('坐标值')
    ax2.set_ylabel('频次')
    ax2.set_title('坐标分布统计', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存汇总图像
    summary_path = os.path.join(output_dir, "analysis_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"汇总图像已保存: analysis_summary.png")

if __name__ == "__main__":
    analyze_and_visualize_coordinates()