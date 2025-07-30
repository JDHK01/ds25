import matplotlib.pyplot as plt
import heapq
import numpy as np
from PIL import Image

# 地图参数
rows, cols = 7, 9  # B1~B7, A1~A9

# 加载地图图像
img_path = "your_image.png"  # 修改为你的文件名
image = Image.open(img_path).convert("RGB")
image_np = np.array(image)

# 每个格子的尺寸
cell_h = image_np.shape[0] // rows
cell_w = image_np.shape[1] // cols

# 估价函数（曼哈顿距离）
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* 路径规划
def a_star(start, goal, forbidden):
    start = (start[1], start[0])  # 转换为 (row, col)
    goal = (goal[1], goal[0])
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal:
            return [(c, r) for r, c in path]
        if current in visited:
            continue
        visited.add(current)

        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols:
                if neighbor in forbidden or neighbor in visited:
                    continue
                heapq.heappush(open_set, (
                    g + 1 + heuristic(neighbor, goal), g + 1, neighbor, path + [neighbor]
                ))
    return None

# 自动识别灰色禁飞区格子
def detect_gray_blocks(image_np, threshold=25):
    forbidden = set()
    for r in range(rows):
        for c in range(cols):
            cell = image_np[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            avg = cell.mean(axis=(0,1))
            if abs(avg[0] - avg[1]) < threshold and abs(avg[1] - avg[2]) < threshold:
                if 90 < avg[0] < 180:  # 中灰
                    forbidden.add((r, c))
    return forbidden

# 起点 (A9)、终点 (A1) = (8,0), (0,6)
start = (8, 6)
goal = (0, 0)

# 检测禁飞区
forbidden = detect_gray_blocks(image_np)

# 路径规划
path = a_star(start, goal, forbidden)

# 可视化结果
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(image_np)

# 网格线
for r in range(1, rows):
    ax.axhline(r * cell_h, color='black', linewidth=0.5)
for c in range(1, cols):
    ax.axvline(c * cell_w, color='black', linewidth=0.5)

# 可视化禁飞区
for (r, c) in forbidden:
    rect = plt.Rectangle((c*cell_w, r*cell_h), cell_w, cell_h,
                         edgecolor='red', facecolor='gray', alpha=0.5)
    ax.add_patch(rect)

# 路径可视化
if path:
    for i in range(len(path)-1):
        x1 = path[i][0] * cell_w + cell_w // 2
        y1 = path[i][1] * cell_h + cell_h // 2
        x2 = path[i+1][0] * cell_w + cell_w // 2
        y2 = path[i+1][1] * cell_h + cell_h // 2
        ax.plot([x1, x2], [y1, y2], 'red', linewidth=3)

    # 起点终点
    sx = start[0] * cell_w + cell_w // 2
    sy = start[1] * cell_h + cell_h // 2
    gx = goal[0] * cell_w + cell_w // 2
    gy = goal[1] * cell_h + cell_h // 2
    ax.plot(sx, sy, 'bo', markersize=10, label='Start')
    ax.plot(gx, gy, 'go', markersize=10, label='Goal')
    ax.legend()
else:
    ax.set_title("未找到可行路径", fontsize=14)

ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.show()
