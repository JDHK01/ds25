import matplotlib.pyplot as plt
from collections import deque
import os
from datetime import datetime

ROWS, COLS = 7, 9
START = (6, 8)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右

def to_custom_label(x, y):
    col = f"A{y+1}"
    row = f"B{ROWS-x}"
    return f"{col}{row}"

def from_custom_label(label):
    try:
        col = int(label[1]) - 1
        row = int(label[3]) - 1
        x = ROWS - 1 - row
        y = col
        if not (0 <= x < ROWS and 0 <= y < COLS):
            raise ValueError
        return (x, y)
    except Exception:
        raise ValueError(f"坐标格式错误: {label}")

def is_valid(x, y, forbidden):
    return 0 <= x < ROWS and 0 <= y < COLS and (x, y) not in forbidden

def traverse_with_obstacles(forbidden_labels):
    """
    根据给定的障碍点列表进行路径遍历
    
    Args:
        forbidden_labels: 障碍点列表，如["A1B1", "A1B2", "A1B3"]
    
    Returns:
        tuple: (path, forbidden, fx_min, fy_min, fx_max, fy_max)
    """
    if len(forbidden_labels) != 3:
        raise ValueError("必须提供恰好3个障碍点")
    
    # 验证障碍点格式并转换为坐标
    forbidden_positions = []
    fx_min = fy_min = 100
    fx_max = fy_max = -1
    
    for label in forbidden_labels:
        label = label.strip().upper()
        pos = from_custom_label(label)
        
        if pos == START:
            raise ValueError(f"障碍点 {label} 不能与起点重合")
        
        if pos in forbidden_positions:
            raise ValueError(f"障碍点 {label} 重复")
        
        forbidden_positions.append(pos)
        fx_min = min(fx_min, pos[0])
        fy_min = min(fy_min, pos[1])
        fx_max = max(fx_max, pos[0])
        fy_max = max(fy_max, pos[1])
    
    forbidden = set(forbidden_positions)
    visited = set()
    path = []

    def dfs(x, y):
        visited.add((x, y))
        path.append((x, y))
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, forbidden) and (nx, ny) not in visited:
                dfs(nx, ny)

    dfs(*START)
    path.append(START)  # 回到起点
    return path, forbidden, fx_min, fy_min, fx_max, fy_max

def find_shortest_path(a, b, forbidden):
    """BFS找a到b的最短路径（不含a，含b），只走上下左右，不经过禁区"""
    queue = deque()
    queue.append((a, []))
    visited = set()
    visited.add(a)
    while queue:
        (x, y), curr_path = queue.popleft()
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            np = (nx, ny)
            if not is_valid(nx, ny, forbidden) or np in visited:
                continue
            if np == b:
                return curr_path + [np]
            queue.append((np, curr_path + [np]))
            visited.add(np)
    return None  # 不可达

def check_and_fix_path(path, fx_min, fy_min, fx_max, fy_max):
    """斜线改折线，直线经过禁区改蛇形线"""
    fixed_path = [path[0]]
    for i in range(1, len(path)):
        x0, y0 = path[i-1]
        x1, y1 = path[i]
        if x0!=x1 and y0!=y1:  # 斜线
            fixed_path.append((x0, y1))
            y0 = y1  # 更新y0为y1
        if fx_max == fx_min:#横禁区
            if x0 == x1 and x0 == fx_min and min(y0,y1) < fy_min and max(y0,y1) > fy_max:#横穿横禁区
                if x0-1<0:
                    fixed_path.append((x0+1, y0))
                    fixed_path.append((x0+1, y1))
                else:
                    fixed_path.append((x0-1, y0))
                    fixed_path.append((x0-1, y1))
            elif y0 == y1 and min(x0,x1) < fx_min and max(x0,x1) > fx_max:  # 竖穿横禁区
                if fy_min-1<0:
                    fixed_path.append((x0, fy_max+1))
                    fixed_path.append((x1, fy_max+1))
                else:
                    fixed_path.append((x0, fy_min-1))
                    fixed_path.append((x1, fy_min-1))
            else:
                pass  # 没有横禁区
        elif fy_max == fy_min:  # 竖禁区
            if y0 == y1 and y0 == fy_min and min(x0,x1) < fx_min and max(x0,x1) > fx_max:  # 竖穿竖禁区
                if y0-1<0:
                    fixed_path.append((x0, y0+1))
                    fixed_path.append((x1, y0+1))
                else:
                    fixed_path.append((x0, y0-1))
                    fixed_path.append((x1, y0-1))
            elif x0 == x1 and min(y0, y1) < fy_min and max(y0, y1) > fy_max:  # 横穿竖禁区
                if fx_min-1<0:
                    fixed_path.append((fx_max+1, y0))
                    fixed_path.append((fx_max+1, y1))
                else:
                    fixed_path.append((fx_min-1, y0))
                    fixed_path.append((fx_min-1, y1))
            else:
                pass  # 没有竖禁区
        fixed_path.append((x1, y1))
    return fixed_path

def draw_grid_path(path, forbidden, save_path=None):
    """
    绘制网格路径图
    
    Args:
        path: 路径点列表
        forbidden: 障碍点集合
        save_path: 图片保存路径，如果为None则不保存
    """
    fig, ax = plt.subplots(figsize=(COLS, ROWS))
    
    # 绘制网格
    for i in range(ROWS):
        for j in range(COLS):
            color = 'white'
            if (i, j) == START:
                color = 'lime'
            elif (i, j) in forbidden:
                color = 'black'
            ax.add_patch(plt.Rectangle((j, ROWS-1-i), 1, 1, edgecolor='gray', facecolor=color))
            ax.text(j+0.5, ROWS-1-i+0.8, to_custom_label(i, j), ha='center', va='top', fontsize=7, color='black')
    
    # 绘制路径箭头
    for idx in range(1, len(path)):
        x0, y0 = path[idx-1][1]+0.5, ROWS-1-path[idx-1][0]+0.5
        x1, y1 = path[idx][1]+0.5, ROWS-1-path[idx][0]+0.5
        ax.arrow(x0, y0, x1-x0, y1-y0, head_width=0.2, length_includes_head=True, color='blue')
    
    # 添加步骤编号
    for k, (i, j) in enumerate(path):
        ax.text(j+0.5, ROWS-1-i+0.5, str(k), ha='center', va='center', fontsize=7, color='red')
    
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.title("遍历路径（绿色为起点，黑色为障碍）")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()

def save_path_to_file(path, filename):
    """
    将路径保存到文件
    
    Args:
        path: 路径点列表
        filename: 保存的文件名
    """
    with open(filename, "w", encoding="utf-8") as f:
        for p in path:
            f.write(to_custom_label(*p) + "\n")
    print(f"路径已保存到: {filename}")

def plan_path_api(obstacle_list, output_dir="output"):
    """
    路径规划API接口
    
    Args:
        obstacle_list: 障碍点列表，如["A1B1", "A1B2", "A1B3"]
        output_dir: 输出目录
    
    Returns:
        dict: 包含路径信息的字典
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用中间参数作为文件名
        middle_param = obstacle_list[1]  # 中间的那个参数
        
        # 执行路径规划
        print(f"开始规划路径，障碍点: {obstacle_list}")
        path, forbidden, fx_min, fy_min, fx_max, fy_max = traverse_with_obstacles(obstacle_list)
        
        # 修复路径
        fixed_path = check_and_fix_path(path, fx_min, fy_min, fx_max, fy_max)
        
        # 转换路径为标签格式
        path_labels = [to_custom_label(*p) for p in fixed_path]
        
        # 保存路径到文件
        path_file = os.path.join(output_dir, f"path_{middle_param}.txt")
        save_path_to_file(fixed_path, path_file)
        
        # 生成并保存图片（使用中间参数命名）
        image_file = os.path.join(output_dir, f"{middle_param}.png")
        draw_grid_path(fixed_path, forbidden, image_file)
        
        # 返回结果
        result = {
            "success": True,
            "obstacle_points": obstacle_list,
            "path_length": len(fixed_path),
            "path_labels": path_labels,
            "path_coordinates": fixed_path,
            "files": {
                "path_file": path_file,
                "image_file": image_file
            },
            "message": f"路径规划成功，共{len(fixed_path)}步"
        }
        
        print(f"路径规划完成！共{len(fixed_path)}步，图片保存为: {middle_param}.png")
        return result
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "message": f"路径规划失败: {e}"
        }
        print(f"错误: {e}")
        return error_result

def process_all_cases():
    """
    处理所有障碍点组合情况并保存图像
    """
    # 所有障碍点组合
    all_obstacle_combinations = [
        # 水平线性障碍（横向）
        ["A1B1", "A1B2", "A1B3"],
        ["A1B2", "A1B3", "A1B4"],
        ["A1B3", "A1B4", "A1B5"],
        ["A1B4", "A1B5", "A1B6"],
        ["A1B5", "A1B6", "A1B7"],
        
        ["A2B1", "A2B2", "A2B3"],
        ["A2B2", "A2B3", "A2B4"],
        ["A2B3", "A2B4", "A2B5"],
        ["A2B4", "A2B5", "A2B6"],
        ["A2B5", "A2B6", "A2B7"],
        
        ["A3B1", "A3B2", "A3B3"],
        ["A3B2", "A3B3", "A3B4"],
        ["A3B3", "A3B4", "A3B5"],
        ["A3B4", "A3B5", "A3B6"],
        ["A3B5", "A3B6", "A3B7"],
        
        ["A4B1", "A4B2", "A4B3"],
        ["A4B2", "A4B3", "A4B4"],
        ["A4B3", "A4B4", "A4B5"],
        ["A4B4", "A4B5", "A4B6"],
        ["A4B5", "A4B6", "A4B7"],
        
        ["A5B1", "A5B2", "A5B3"],
        ["A5B2", "A5B3", "A5B4"],
        ["A5B3", "A5B4", "A5B5"],
        ["A5B4", "A5B5", "A5B6"],
        ["A5B5", "A5B6", "A5B7"],
        
        ["A6B1", "A6B2", "A6B3"],
        ["A6B2", "A6B3", "A6B4"],
        ["A6B3", "A6B4", "A6B5"],
        ["A6B4", "A6B5", "A6B6"],
        ["A6B5", "A6B6", "A6B7"],
        
        ["A7B1", "A7B2", "A7B3"],
        ["A7B2", "A7B3", "A7B4"],
        ["A7B3", "A7B4", "A7B5"],
        ["A7B4", "A7B5", "A7B6"],
        ["A7B5", "A7B6", "A7B7"],
        
        ["A8B1", "A8B2", "A8B3"],
        ["A8B2", "A8B3", "A8B4"],
        ["A8B3", "A8B4", "A8B5"],
        ["A8B4", "A8B5", "A8B6"],
        ["A8B5", "A8B6", "A8B7"],
        
        ["A9B2", "A9B3", "A9B4"],
        ["A9B3", "A9B4", "A9B5"],
        ["A9B4", "A9B5", "A9B6"],
        ["A9B5", "A9B6", "A9B7"],
        
        # 垂直线性障碍（纵向）
        ["A1B1", "A2B1", "A3B1"],
        ["A2B1", "A3B1", "A4B1"],
        ["A3B1", "A4B1", "A5B1"],
        ["A4B1", "A5B1", "A6B1"],
        ["A5B1", "A6B1", "A7B1"],
        ["A6B1", "A7B1", "A8B1"],
        
        ["A1B2", "A2B2", "A3B2"],
        ["A2B2", "A3B2", "A4B2"],
        ["A3B2", "A4B2", "A5B2"],
        ["A4B2", "A5B2", "A6B2"],
        ["A5B2", "A6B2", "A7B2"],
        ["A6B2", "A7B2", "A8B2"],
        ["A7B2", "A8B2", "A9B2"],
        
        ["A1B3", "A2B3", "A3B3"],
        ["A2B3", "A3B3", "A4B3"],
        ["A3B3", "A4B3", "A5B3"],
        ["A4B3", "A5B3", "A6B3"],
        ["A5B3", "A6B3", "A7B3"],
        ["A6B3", "A7B3", "A8B3"],
        ["A7B3", "A8B3", "A9B3"],
        
        ["A1B4", "A2B4", "A3B4"],
        ["A2B4", "A3B4", "A4B4"],
        ["A3B4", "A4B4", "A5B4"],
        ["A4B4", "A5B4", "A6B4"],
        ["A5B4", "A6B4", "A7B4"],
        ["A6B4", "A7B4", "A8B4"],
        ["A7B4", "A8B4", "A9B4"],
        
        ["A1B5", "A2B5", "A3B5"],
        ["A2B5", "A3B5", "A4B5"],
        ["A3B5", "A4B5", "A5B5"],
        ["A4B5", "A5B5", "A6B5"],
        ["A5B5", "A6B5", "A7B5"],
        ["A6B5", "A7B5", "A8B5"],
        ["A7B5", "A8B5", "A9B5"],
        
        ["A1B6", "A2B6", "A3B6"],
        ["A2B6", "A3B6", "A4B6"],
        ["A3B6", "A4B6", "A5B6"],
        ["A4B6", "A5B6", "A6B6"],
        ["A5B6", "A6B6", "A7B6"],
        ["A6B6", "A7B6", "A8B6"],
        ["A7B6", "A8B6", "A9B6"],
        
        ["A1B7", "A2B7", "A3B7"],
        ["A2B7", "A3B7", "A4B7"],
        ["A3B7", "A4B7", "A5B7"],
        ["A4B7", "A5B7", "A6B7"],
        ["A5B7", "A6B7", "A7B7"],
        ["A6B7", "A7B7", "A8B7"],
        ["A7B7", "A8B7", "A9B7"]
    ]
    
    print(f"开始处理 {len(all_obstacle_combinations)} 种障碍点组合...")
    
    successful_cases = 0
    failed_cases = 0
    
    for i, obstacles in enumerate(all_obstacle_combinations, 1):
        print(f"\n处理第 {i}/{len(all_obstacle_combinations)} 种情况: {obstacles}")
        
        try:
            result = plan_path_api(obstacles)
            
            if result["success"]:
                successful_cases += 1
                print(f"✓ 成功 - 图片保存为: {obstacles[1]}.png")
            else:
                failed_cases += 1
                print(f"✗ 失败: {result['message']}")
                
        except Exception as e:
            failed_cases += 1
            print(f"✗ 异常: {e}")
    
    print(f"\n=== 处理完成 ===")
    print(f"成功: {successful_cases} 种情况")
    print(f"失败: {failed_cases} 种情况")
    print(f"总计: {len(all_obstacle_combinations)} 种情况")

# 使用示例
if __name__ == '__main__':
    # 处理所有情况
    process_all_cases()
    
    # 也可以单独测试一种情况
    # obstacles = ["A1B2", "A1B3", "A1B4"]
    # result = plan_path_api(obstacles)
    # 
    # if result["success"]:
    #     print("\n=== 规划结果 ===")
    #     print(f"障碍点: {result['obstacle_points']}")
    #     print(f"路径长度: {result['path_length']}")
    #     print(f"完整路径: {' -> '.join(result['path_labels'])}")
    #     print(f"文件保存位置: {result['files']}")
    # else:
    #     print(f"规划失败: {result['message']}")