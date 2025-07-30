import matplotlib.pyplot as plt
from collections import deque

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

def find_shortest_path(a, b, forbidden):
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
    return None

def check_and_fix_path(path, forbidden):
    fixed_path = [path[0]]
    for i in range(1, len(path)):
        prev = fixed_path[-1]
        curr = path[i]
        if abs(prev[0] - curr[0]) + abs(prev[1] - curr[1]) == 1:
            fixed_path.append(curr)
        else:
            sub_path = find_shortest_path(prev, curr, forbidden)
            if sub_path is None:
                continue
            fixed_path.extend(sub_path)
    return fixed_path

def draw_grid_path(path, forbidden):
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
    for idx in range(1, len(path)):
        x0, y0 = path[idx-1][1]+0.5, ROWS-1-path[idx-1][0]+0.5
        x1, y1 = path[idx][1]+0.5, ROWS-1-path[idx][0]+0.5
        ax.arrow(x0, y0, x1-x0, y1-y0, head_width=0.2, length_includes_head=True, color='blue')
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

def plan_path(input_labels, visualize=False):
    """
    封装主逻辑：
    输入：input_labels 为无序的三个障碍点标签（如 ["A1B7", "A3B6", "A2B4"]）
    输出：返回路径坐标列表
    参数：visualize=True 时开启图形化显示路径
    """
    if len(input_labels) != 3:
        raise ValueError("输入必须是3个坐标标签。")

    forbidden = set()
    for label in input_labels:
        pos = from_custom_label(label)
        if pos == START:
            raise ValueError(f"障碍点 {label} 不能与起点重合。")
        forbidden.add(pos)

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
    path.append(START)
    path = check_and_fix_path(path, forbidden)

    if visualize:
        draw_grid_path(path, forbidden)

    return path

if __name__ == '__main__':
    input_data = ["A1B6", "A1B5", "A1B4"]
    result_path = plan_path(input_data, visualize=True)  # 设置为 False 可关闭图形
    for step in result_path:
        print(to_custom_label(*step))
