import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def parse_coordinate(coord):
    """解析坐标格式 'AnBm' 返回 (n-1, m-1) 作为数组索引"""
    # 提取A后面的数字和B后面的数字
    a_pos = coord.find('A')
    b_pos = coord.find('B')
    
    if a_pos == -1 or b_pos == -1:
        raise ValueError(f"无效的坐标格式: {coord}")
    
    x = int(coord[a_pos+1:b_pos]) - 1  # A1->0, A2->1, ...
    y = int(coord[b_pos+1:]) - 1       # B1->0, B2->1, ...
    
    return x, y

def visualize_grid_navigation(gray_cells, route_path, grid_size=(9, 7)):
    """
    可视化方格导航图
    
    参数:
    gray_cells: 需要标灰的方格列表，如 ["A1B3","A1B4","A1B5"]
    route_path: 航线路径列表，如 ["A9B1","A9B2",...]
    grid_size: 网格大小 (宽度, 高度)
    """
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 设置网格范围
    width, height = grid_size
    
    # 绘制网格线
    for i in range(width + 1):
        ax.axvline(x=i, color='gray', linewidth=0.8, alpha=0.6)
    for i in range(height + 1):
        ax.axhline(y=i, color='gray', linewidth=0.8, alpha=0.6)
    
    # 添加坐标标签
    # 列标签 (A1, A2, ..., A9)
    for i in range(width):
        ax.text(i + 0.5, height + 0.2, f'A{i+1}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 行标签 (B7, B6, ..., B1) - 注意Y轴是从下往上的
    for i in range(height):
        ax.text(-0.2, height - i - 0.5, f'B{i+1}', ha='right', va='center', fontsize=10, fontweight='bold')
    
    # 标记灰色方格
    for cell in gray_cells:
        x, y = parse_coordinate(cell)
        # 注意：matplotlib的y轴需要翻转，因为我们的B1在底部
        rect = patches.Rectangle((x, height - y - 1), 1, 1, 
                               linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.6)
        ax.add_patch(rect)
    
    # 绘制航线路径
    if route_path:
        # 解析所有路径点
        path_points = []
        for point in route_path:
            x, y = parse_coordinate(point)
            # 转换为方格中心坐标，并翻转Y轴
            center_x = x + 0.5
            center_y = height - y - 0.5
            path_points.append((center_x, center_y))
        
        # 绘制路径线
        if len(path_points) > 1:
            path_x = [p[0] for p in path_points]
            path_y = [p[1] for p in path_points]
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='航线路径')
            
            # 添加箭头指示方向
            for i in range(len(path_points) - 1):
                dx = path_points[i+1][0] - path_points[i][0]
                dy = path_points[i+1][1] - path_points[i][1]
                if dx != 0 or dy != 0:  # 避免零长度箭头
                    ax.annotate('', xy=path_points[i+1], xytext=path_points[i],
                              arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5, lw=1))
        
        # 标记起点（红色方格和白色圆点）
        if path_points:
            start_x, start_y = path_points[0]
            # 红色方格
            start_rect = patches.Rectangle((start_x - 0.5, start_y - 0.5), 1, 1,
                                         linewidth=2, edgecolor='darkred', facecolor='red', alpha=0.8)
            ax.add_patch(start_rect)
            # 白色圆点
            circle = patches.Circle((start_x, start_y), 0.15, color='white', zorder=5)
            ax.add_patch(circle)
        
        # 标记终点
        if len(path_points) > 1:
            end_x, end_y = path_points[-1]
            end_circle = patches.Circle((end_x, end_y), 0.2, color='green', zorder=5)
            ax.add_patch(end_circle)
    
    # 设置坐标轴
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_title('方格航线导航图', fontsize=14, fontweight='bold')
    
    # 移除刻度
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 添加图例
    legend_elements = [
        patches.Patch(color='lightgray', alpha=0.6, label='障碍区域'),
        plt.Line2D([0], [0], color='blue', linewidth=2, alpha=0.7, label='航线路径'),
        patches.Patch(color='red', alpha=0.8, label='起点'),
        patches.Patch(color='green', label='终点')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    return fig, ax

# 示例使用
if __name__ == "__main__":
    # 灰色方格（障碍物）
    gray_cells = ["A7B7", "A8B7", "A9B7"]
    
    # 航线路径
    route_path = ['A9B1', 'A9B2', 'A9B3', 'A9B4', 'A9B5', 'A9B6', 'A8B6', 'A8B5', 'A8B4', 'A8B3', 'A8B2', 'A7B2', 'A7B3', 'A7B4', 'A7B5', 'A7B6', 'A6B6', 'A6B7', 'A5B7', 'A5B6', 'A5B5', 'A6B5', 'A6B4', 'A6B3', 'A6B2', 'A5B2', 'A5B3', 'A5B4', 'A4B4', 'A4B5', 'A4B6', 'A4B7', 'A3B7', 'A3B6', 'A3B5', 'A3B4', 'A3B3', 'A4B3', 'A4B2', 'A3B2', 'A2B2', 'A2B3', 'A2B4', 'A2B5', 'A2B6', 'A2B7', 'A1B7', 'A1B6', 'A1B5', 'A1B4', 'A1B3', 'A1B2', 'A1B1', 'A2B1', 'A3B1', 'A4B1', 'A5B1', 'A6B1', 'A7B1', 'A8B1', 'A9B1']
    
    # 创建可视化
    fig, ax = visualize_grid_navigation(gray_cells, route_path)
    
    # 显示图形
    plt.show()
    
    # 如果需要保存图片
    # plt.savefig('grid_navigation.png', dpi=300, bbox_inches='tight')

# # 额外功能：分析航线
# def analyze_route(route_path):
#     """分析航线特征"""
#     if not route_path:
#         return
    
#     print(f"航线分析:")
#     print(f"总步数: {len(route_path)}")
#     print(f"起点: {route_path[0]}")
#     print(f"终点: {route_path[-1]}")
    
#     # 计算移动方向统计
#     directions = {'上': 0, '下': 0, '左': 0, '右': 0}
    
#     for i in range(len(route_path) - 1):
#         x1, y1 = parse_coordinate(route_path[i])
#         x2, y2 = parse_coordinate(route_path[i + 1])
        
#         if x2 > x1:
#             directions['右'] += 1
#         elif x2 < x1:
#             directions['左'] += 1
#         elif y2 > y1:
#             directions['上'] += 1
#         elif y2 < y1:
#             directions['下'] += 1
    
#     print("移动方向统计:")
#     for direction, count in directions.items():
#         print(f"  {direction}: {count}步")

# # 运行航线分析
# if __name__ == "__main__":
#     route_path = [
#         "A9B1","A9B2","A9B3","A9B4","A9B5","A9B6","A9B7","A8B7","A7B7","A6B7",
#         "A5B7","A4B7","A3B7","A2B7","A1B7","A1B6","A2B6","A2B5","A2B4","A2B3",
#         "A2B2","A1B2","A1B1","A2B1","A3B1","A3B2","A3B3","A3B4","A3B5","A3B6",
#         "A4B6","A4B5","A4B4","A4B3","A4B2","A4B1","A5B1","A5B2","A5B3","A5B4",
#         "A5B5","A5B6","A6B6","A6B5","A6B4","A6B3","A6B2","A6B1","A7B1","A7B2",
#         "A7B3","A7B4","A7B5","A7B6","A8B6","A8B5","A8B4","A8B3","A8B2","A8B1","A9B1"
#     ]
    
#     analyze_route(route_path)