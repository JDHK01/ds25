import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time
from collections import deque

ROWS, COLS = 7, 9
BARRIER_COUNT = 3
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class Point:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __hash__(self):
        return hash((self.row, self.col))

def init_grid():
    return [[1 for _ in range(COLS)] for _ in range(ROWS)]

def is_valid(row, col):
    return 0 <= row < ROWS and 0 <= col < COLS

def manhattan_distance(a, b):
    return abs(a.row - b.row) + abs(a.col - b.col)

def generate_barriers(grid):
    for _ in range(100):
        temp_grid = init_grid()
        direction = random.randint(0, 1)

        if direction == 0:  # 水平
            row = random.randint(0, ROWS - 1)
            col = random.randint(0, COLS - 3)
            if row == 0 and col <= 0 <= col + 2:
                continue
            for i in range(3):
                temp_grid[row][col + i] = 0
        else:  # 垂直
            col = random.randint(0, COLS - 1)
            row = random.randint(0, ROWS - 3)
            if col == 0 and row <= 0 <= row + 2:
                continue
            for i in range(3):
                temp_grid[row + i][col] = 0

        if check_connectivity(temp_grid):
            return temp_grid
    # 默认障碍
    grid[2][3] = grid[2][4] = grid[2][5] = 0
    return grid

def check_connectivity(grid):
    visited = [[False]*COLS for _ in range(ROWS)]
    q = deque()
    q.append(Point(0, 0))
    visited[0][0] = True
    count = 1

    while q:
        p = q.popleft()
        for d in directions:
            nr, nc = p.row + d[0], p.col + d[1]
            if is_valid(nr, nc) and grid[nr][nc] == 1 and not visited[nr][nc]:
                visited[nr][nc] = True
                q.append(Point(nr, nc))
                count += 1

    accessible = sum(row.count(1) for row in grid)
    return count >= accessible * 0.8

def collect_accessible(grid):
    return [Point(i, j) for i in range(ROWS) for j in range(COLS) if grid[i][j] == 1]

def bfs_path(grid, start, end):
    visited = [[False]*COLS for _ in range(ROWS)]
    parent = {}
    q = deque()
    q.append(start)
    visited[start.row][start.col] = True
    parent[start] = None

    while q:
        cur = q.popleft()
        if cur == end:
            path = []
            while cur:
                path.append(cur)
                cur = parent[cur]
            return list(reversed(path))
        for d in directions:
            nr, nc = cur.row + d[0], cur.col + d[1]
            nxt = Point(nr, nc)
            if is_valid(nr, nc) and grid[nr][nc] == 1 and not visited[nr][nc]:
                visited[nr][nc] = True
                parent[nxt] = cur
                q.append(nxt)
    return []

def tsp_nearest(grid, accessible):
    visited = set()
    order = []
    current = Point(0, 0)
    visited.add(current)
    order.append(current)

    while len(visited) < len(accessible):
        next_point = None
        min_dist = float('inf')
        for p in accessible:
            if p not in visited:
                dist = manhattan_distance(current, p)
                if dist < min_dist:
                    min_dist = dist
                    next_point = p
        if next_point:
            visited.add(next_point)
            order.append(next_point)
            current = next_point
        else:
            break
    order.append(Point(0, 0))  # 返回起点
    return order

def draw_grid(grid, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_xticks(range(COLS+1))
    ax.set_yticks(range(ROWS+1))
    ax.grid(True)
    ax.set_title("路径可视化")

    for i in range(ROWS):
        for j in range(COLS):
            if grid[i][j] == 0:
                ax.add_patch(patches.Rectangle((j, ROWS-i-1), 1, 1, color='black'))

    for i, point in enumerate(path):
        row = point.row
        col = point.col
        color = 'red' if i == 0 else ('blue' if i == len(path)-1 else 'green')
        ax.add_patch(patches.Circle((col + 0.5, ROWS - row - 0.5), 0.2, color=color))
        if i > 0:
            prev = path[i - 1]
            ax.annotate('', xy=(col + 0.5, ROWS - row - 0.5),
                        xytext=(prev.col + 0.5, ROWS - prev.row - 0.5),
                        arrowprops=dict(arrowstyle='->', color='orange'))

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def solve():
    grid = init_grid()
    grid = generate_barriers(grid)
    accessible = collect_accessible(grid)
    order = tsp_nearest(grid, accessible)

    full_path = []
    for i in range(len(order) - 1):
        segment = bfs_path(grid, order[i], order[i + 1])
        if i > 0:
            segment = segment[1:]  # 避免重复
        full_path.extend(segment)

    print("\n访问顺序:")
    for i, p in enumerate(order):
        print(f"({p.row+1},{p.col+1})", end=" -> " if i < len(order)-1 else "\n")

    print(f"\n总步数: {len(full_path)-1}")
    print(f"可访问格子数: {len(accessible)}")
    print(f"路径效率: {len(accessible)/len(full_path)*100:.1f}%")
    draw_grid(grid, full_path)

if __name__ == '__main__':
    solve()
