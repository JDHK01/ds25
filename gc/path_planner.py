import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from collections import deque
import random
import time

class AdvancedWildlifePatrolSystem:
    def __init__(self):
        self.rows = 7
        self.cols = 9
        self.grid = np.zeros((self.rows, self.cols))
        self.forbidden_zones = []
        self.path = []
        self.start_pos = (8, 0)  # A9B1
        self.all_paths = []  # 存储多条路径用于比较
        
    def set_forbidden_zone(self, start_col, start_row, shape='horizontal'):
        """设置禁飞区"""
        self.forbidden_zones = []
        self.grid = np.zeros((self.rows, self.cols))
        
        if shape == 'horizontal' and start_col <= 6:
            for i in range(3):
                self.forbidden_zones.append((start_col + i, start_row))
                self.grid[start_row, start_col + i] = 1
        elif shape == 'vertical' and start_row <= 4:
            for i in range(3):
                self.forbidden_zones.append((start_col, start_row + i))
                self.grid[start_row + i, start_col] = 1
    
    def get_accessible_points(self):
        """获取所有可访问的点"""
        points = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row, col] == 0:
                    points.append((col, row))
        return points
    
    def get_neighbors(self, pos):
        """获取某个位置的有效邻居"""
        col, row = pos
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dx, dy in directions:
            new_col, new_row = col + dx, row + dy
            if (0 <= new_col < self.cols and 
                0 <= new_row < self.rows and 
                self.grid[new_row, new_col] == 0):
                neighbors.append((new_col, new_row))
        
        return neighbors
    
    def manhattan_distance(self, p1, p2):
        """计算曼哈顿距离"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def plan_path_greedy_enhanced(self):
        """增强贪心算法 - 考虑未来可达性"""
        accessible = self.get_accessible_points()
        visited = set()
        path = [self.start_pos]
        visited.add(self.start_pos)
        current = self.start_pos
        
        while len(visited) < len(accessible):
            neighbors = [n for n in self.get_neighbors(current) if n not in visited]
            
            if neighbors:
                # 评估每个邻居
                best_neighbor = None
                best_score = -float('inf')
                
                for neighbor in neighbors:
                    # 计算选择这个邻居后的得分
                    score = 0
                    
                    # 1. 基础得分：未访问邻居的数量
                    temp_visited = visited.copy()
                    temp_visited.add(neighbor)
                    
                    for point in accessible:
                        if point not in temp_visited:
                            point_neighbors = [n for n in self.get_neighbors(point) 
                                             if n not in temp_visited]
                            score += len(point_neighbors)
                    
                    # 2. 避免死角：检查是否会创建孤立点
                    isolated_penalty = 0
                    for point in accessible:
                        if point not in temp_visited:
                            point_neighbors = [n for n in self.get_neighbors(point) 
                                             if n not in temp_visited]
                            if len(point_neighbors) == 0:
                                isolated_penalty += 100
                    score -= isolated_penalty
                    
                    # 3. 最后阶段优先靠近起点
                    if len(visited) >= len(accessible) - 5:
                        dist_to_start = self.manhattan_distance(neighbor, self.start_pos)
                        score += (20 - dist_to_start) * 5
                    
                    if score > best_score:
                        best_score = score
                        best_neighbor = neighbor
                
                current = best_neighbor
            else:
                # 寻找最近的未访问点
                min_dist = float('inf')
                nearest = None
                for point in accessible:
                    if point not in visited:
                        dist = self.manhattan_distance(point, current)
                        if dist < min_dist:
                            min_dist = dist
                            nearest = point
                
                if nearest:
                    current = nearest
                else:
                    break
            
            path.append(current)
            visited.add(current)
        
        # 回到起点
        path.append(self.start_pos)
        return path
    
    def plan_path_spiral(self):
        """螺旋式路径规划"""
        accessible = set(self.get_accessible_points())
        visited = set()
        path = [self.start_pos]
        visited.add(self.start_pos)
        
        # 定义螺旋方向：右、上、左、下
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dir_idx = 0
        current = self.start_pos
        
        while len(visited) < len(accessible):
            moved = False
            
            # 尝试按当前方向移动
            for _ in range(4):  # 最多尝试4个方向
                dx, dy = directions[dir_idx]
                next_pos = (current[0] + dx, current[1] + dy)
                
                if (next_pos in accessible and 
                    next_pos not in visited and 
                    0 <= next_pos[0] < self.cols and 
                    0 <= next_pos[1] < self.rows):
                    current = next_pos
                    path.append(current)
                    visited.add(current)
                    moved = True
                    break
                else:
                    # 转向
                    dir_idx = (dir_idx + 1) % 4
            
            # 如果螺旋被阻挡，找最近的未访问点
            if not moved:
                min_dist = float('inf')
                nearest = None
                for point in accessible:
                    if point not in visited:
                        dist = self.manhattan_distance(point, current)
                        if dist < min_dist:
                            min_dist = dist
                            nearest = point
                
                if nearest:
                    current = nearest
                    path.append(current)
                    visited.add(current)
                else:
                    break
        
        path.append(self.start_pos)
        return path
    
    def plan_path_zigzag(self):
        """之字形路径规划"""
        accessible = self.get_accessible_points()
        visited = set()
        path = [self.start_pos]
        visited.add(self.start_pos)
        
        # 按行组织点
        rows_dict = {}
        for point in accessible:
            row = point[1]
            if row not in rows_dict:
                rows_dict[row] = []
            rows_dict[row].append(point)
        
        # 对每行的点按列排序
        for row in rows_dict:
            rows_dict[row].sort(key=lambda x: x[0])
        
        # 从起点所在行开始
        current_row = self.start_pos[1]
        direction = 1  # 1: 向右, -1: 向左
        
        # 访问当前行
        if current_row in rows_dict:
            row_points = rows_dict[current_row]
            if direction == 1:
                for point in row_points:
                    if point not in visited:
                        path.append(point)
                        visited.add(point)
            else:
                for point in reversed(row_points):
                    if point not in visited:
                        path.append(point)
                        visited.add(point)
        
        # 访问其他行
        for row in range(self.rows):
            if row != current_row and row in rows_dict:
                direction *= -1  # 改变方向
                row_points = rows_dict[row]
                
                if direction == 1:
                    for point in row_points:
                        if point not in visited:
                            path.append(point)
                            visited.add(point)
                else:
                    for point in reversed(row_points):
                        if point not in visited:
                            path.append(point)
                            visited.add(point)
        
        # 确保访问所有点
        for point in accessible:
            if point not in visited:
                path.append(point)
                visited.add(point)
        
        path.append(self.start_pos)
        return path
    
    def evaluate_path(self, path):
        """评估路径质量"""
        if len(path) < 2:
            return float('inf')
        
        total_distance = 0
        revisits = 0
        visited = set()
        
        for i in range(len(path) - 1):
            # 计算总距离
            total_distance += self.manhattan_distance(path[i], path[i + 1])
            
            # 统计重复访问
            if path[i] in visited:
                revisits += 1
            visited.add(path[i])
        
        # 检查是否覆盖所有可访问点
        accessible = set(self.get_accessible_points())
        coverage = len(set(path) & accessible) / len(accessible)
        
        # 综合评分（越小越好）
        score = total_distance + revisits * 10 + (1 - coverage) * 100
        
        return score
    
    def plan_optimal_path(self):
        """尝试多种算法，选择最优路径"""
        algorithms = [
            ("增强贪心算法", self.plan_path_greedy_enhanced),
            ("螺旋式算法", self.plan_path_spiral),
            ("之字形算法", self.plan_path_zigzag)
        ]
        
        best_path = None
        best_score = float('inf')
        best_algorithm = ""
        
        print("正在尝试不同的路径规划算法...")
        
        for name, algorithm in algorithms:
            try:
                path = algorithm()
                score = self.evaluate_path(path)
                print(f"{name}: 路径长度={len(path)}, 评分={score:.2f}")
                
                if score < best_score:
                    best_score = score
                    best_path = path
                    best_algorithm = name
            except Exception as e:
                print(f"{name} 失败: {e}")
        
        print(f"\n最优算法: {best_algorithm}")
        self.path = best_path
        return best_path
    
    def visualize_comparison(self):
        """可视化比较不同算法的结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        algorithms = [
            ("增强贪心算法", self.plan_path_greedy_enhanced),
            ("螺旋式算法", self.plan_path_spiral),
            ("之字形算法", self.plan_path_zigzag),
            ("最优路径", lambda: self.path)
        ]
        
        for idx, (ax, (name, algorithm)) in enumerate(zip(axes.flat, algorithms)):
            if name != "最优路径":
                path = algorithm()
            else:
                path = self.path
            
            self._draw_grid(ax, path, name)
        
        plt.tight_layout()
        plt.show()
    
    def _draw_grid(self, ax, path, title):
        """在指定的轴上绘制网格和路径"""
        # 绘制网格
        for i in range(self.cols + 1):
            ax.axvline(x=i, color='black', linewidth=0.5)
        for i in range(self.rows + 1):
            ax.axhline(y=i, color='black', linewidth=0.5)
        
        # 绘制方格标签
        for row in range(self.rows):
            for col in range(self.cols):
                label = f"{chr(65 + col)}{row + 1}"
                ax.text(col + 0.5, row + 0.5, label, 
                       ha='center', va='center', fontsize=8)
        
        # 绘制禁飞区
        for col, row in self.forbidden_zones:
            rect = patches.Rectangle((col, row), 1, 1, 
                                   linewidth=1, edgecolor='black', 
                                   facecolor='gray', alpha=0.7)
            ax.add_patch(rect)
        
        # 绘制起点
        start_rect = patches.Rectangle((self.start_pos[0], self.start_pos[1]), 
                                     1, 1, linewidth=2, edgecolor='black', 
                                     facecolor='red', alpha=0.7)
        ax.add_patch(start_rect)
        
        # 绘制路径
        if path:
            for i, (col, row) in enumerate(path[:-1]):
                if (col, row) != self.start_pos:
                    rect = patches.Rectangle((col, row), 1, 1, 
                                           linewidth=1, edgecolor='black', 
                                           facecolor='lightgreen', alpha=0.5)
                    ax.add_patch(rect)
                
                # 添加序号
                ax.text(col + 0.8, row + 0.2, str(i + 1), 
                       ha='center', va='center', fontsize=6, 
                       color='blue', fontweight='bold')
            
            # 绘制路径线
            for i in range(len(path) - 1):
                start = (path[i][0] + 0.5, path[i][1] + 0.5)
                end = (path[i + 1][0] + 0.5, path[i + 1][1] + 0.5)
                
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       'b-', linewidth=1.5, alpha=0.6)
        
        # 设置标题和属性
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'{title}\n路径长度: {len(path) if path else 0}', fontsize=12)

# 使用示例
if __name__ == "__main__":
    # 创建高级巡查系统
    patrol = AdvancedWildlifePatrolSystem()
    
    # 测试不同的禁飞区配置
    test_cases = [
        ("横向禁飞区 (C4-E4)", 2, 3, 'horizontal'),
        ("纵向禁飞区 (F2-F4)", 5, 1, 'vertical'),
        ("边缘横向禁飞区 (A7-C7)", 0, 6, 'horizontal'),
    ]
    
    for case_name, col, row, shape in test_cases:
        print(f"\n{'='*50}")
        print(f"测试案例: {case_name}")
        print(f"{'='*50}")
        
        patrol.set_forbidden_zone(col, row, shape)
        
        # 找到最优路径
        optimal_path = patrol.plan_optimal_path()
        
        # 打印路径详情
        print("\n最优路径详情:")
        for i, (col, row) in enumerate(optimal_path[:10]):  # 只打印前10步
            cell_name = f"{chr(65 + col)}{row + 1}"
            print(f"步骤 {i + 1}: {cell_name}")
        if len(optimal_path) > 10:
            print(f"... (共 {len(optimal_path)} 步)")
        
        # 可视化比较
        patrol.visualize_comparison()
        
        # 等待用户输入继续
        input("\n按Enter继续下一个测试案例...")