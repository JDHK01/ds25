import matplotlib.pyplot as plt
from collections import deque

ROWS, COLS = 7, 9
START = (6, 8)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右

def to_custom_label(x, y):
    """将内部坐标转换为自定义标签格式"""
    col = f"A{y+1}"
    row = f"B{ROWS-x}"
    return f"{col}{row}"

def from_custom_label(label):
    """将自定义标签格式转换为内部坐标"""
    try:
        col = int(label[1]) - 1
        row = int(label[3]) - 1
        x = ROWS - 1 - row
        y = col
        if not (0 <= x < ROWS and 0 <= y < COLS):
            raise ValueError
        return (x, y)
    except Exception:
        raise ValueError("坐标格式错误")

def is_valid(x, y, forbidden):
    """检查坐标是否有效"""
    return 0 <= x < ROWS and 0 <= y < COLS and (x, y) not in forbidden

def check_and_fix_path(path, forbidden, fx_min, fy_min, fx_max, fy_max):
    """修复路径：斜线改折线，直线经过禁区改蛇形线"""
    fixed_path = [path[0]]
    for i in range(1, len(path)):
        x0, y0 = path[i-1]
        x1, y1 = path[i]
        if x0!=x1 and y0!=y1:  # 斜线
            x2, y2 = path[i]
            x1=x0
            if (x1, y1) in forbidden:
                x1 =x2
                y1= y0
                fixed_path.append((x1, y1))  # 水平线
                x0=x1
            else:
                if fx_max == fx_min:#横禁区
                    if x0 == x1 and x0 == fx_min and min(y0,y1) < fy_min and max(y0,y1) > fy_max:#横穿横禁区
                        if x0-1<0:
                            fixed_path.append((x0+1, y0))
                            fixed_path.append((x0+1, y1))
                        else:
                            fixed_path.append((x0-1, y0))
                            fixed_path.append((x0-1, y1))
                elif fy_max == fy_min:  # 竖禁区
                    if x0 == x1 and min(y0, y1) < fy_min and max(y0, y1) > fy_max:  # 横穿竖禁区
                        if fx_min-1<0:
                            fixed_path.append((fx_max+1, y0))
                            fixed_path.append((fx_max+1, y1))
                        else:
                            fixed_path.append((fx_min-1, y0))
                            fixed_path.append((fx_min-1, y1))
                fixed_path.append((x1, y1))
                y0 = y2  # 更新y0为y1
        x1, y1 = path[i]
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

def plan_path(obstacles):
    """
    路径规划API函数
    
    Args:
        obstacles (list): 障碍物坐标列表，格式如 ['A1B1', 'A1B2', 'A1B3']
    
    Returns:
        list: 规划的航点坐标列表，格式如 ['A9B1', 'A8B1', ...]
    
    Raises:
        ValueError: 当输入格式错误、障碍物与起点重合或超出网格范围时
    """
    # 输入验证
    if not isinstance(obstacles, list):
        raise ValueError("障碍物输入必须是列表格式")
    
    if len(obstacles) != 3:
        raise ValueError("必须输入3个障碍物坐标")
    
    # 转换障碍物坐标并验证
    forbidden = set()
    fx_min = fy_min = 100
    fx_max = fy_max = -1
    
    for obs in obstacles:
        try:
            pos = from_custom_label(obs.strip().upper())
            if pos == START:
                raise ValueError(f"障碍物 {obs} 不能与起点重合")
            if pos in forbidden:
                raise ValueError(f"障碍物 {obs} 重复输入")
            forbidden.add(pos)
            fx_min = min(fx_min, pos[0])
            fy_min = min(fy_min, pos[1])
            fx_max = max(fx_max, pos[0])
            fy_max = max(fy_max, pos[1])
        except ValueError as e:
            if "障碍物" in str(e):
                raise e
            else:
                raise ValueError(f"障碍物坐标 {obs} 格式错误")
    
    # DFS遍历所有可达格子
    visited = set()
    path = []
    
    def dfs(x, y):
        visited.add((x, y))
        path.append((x, y))
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, forbidden) and (nx, ny) not in visited:
                dfs(nx, ny)
    
    # 从起点开始遍历
    dfs(*START)
    path.append(START)  # 回到起点
    
    # 修复路径
    fixed_path = check_and_fix_path(path, forbidden, fx_min, fy_min, fx_max, fy_max)
    
    # 转换为自定义标签格式并返回
    result = [to_custom_label(x, y) for x, y in fixed_path]
    
    return result

def visualize_path(obstacles, path_result=None):
    """
    可视化路径规划结果
    
    Args:
        obstacles (list): 障碍物坐标列表
        path_result (list, optional): 路径结果，如果为None则自动计算
    """
    if path_result is None:
        path_result = plan_path(obstacles)
    
    # 转换为内部坐标
    forbidden = set(from_custom_label(obs.strip().upper()) for obs in obstacles)
    path = [from_custom_label(label) for label in path_result]
    
    # 绘制网格
    fig, ax = plt.subplots(figsize=(COLS, ROWS))
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
    
    # 标记步骤序号
    for k, (i, j) in enumerate(path):
        ax.text(j+0.5, ROWS-1-i+0.5, str(k), ha='center', va='center', fontsize=7, color='red')
    
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.title("遍历路径（绿色为起点，黑色为障碍）")
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == '__main__':
    # 示例用法
    obstacles = ['A1B1', 'A1B2', 'A1B3']
    result = plan_path(obstacles)