import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import tqdm

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 创建保存图像的目录
output_dir = 'grid_patterns'
os.makedirs(output_dir, exist_ok=True)

# 定义颜色映射
colors = [(1, 1, 1), (0.7, 0.7, 0.7)]  # 白色和灰色
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=2)

# 棋盘尺寸
ROWS, COLS = 7, 9

# 初始化计数器
total_count = 0
generated_files = []

# 生成所有可能的横向三连方块
print("正在生成横向三连方块图案...")
for i in tqdm.tqdm(range(ROWS)):
    for j in range(COLS - 2):
        # 创建空白棋盘
        grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        # 标记三个连续方块为灰色
        for k in range(3):
            grid[i][j + k] = 1
        
        # 绘制棋盘
        fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
        ax.imshow(grid, cmap=cmap, interpolation='nearest')
        
        # 调整坐标系统，使方格居中
        ax.set_xticks([x + 0.5 for x in range(COLS)])
        ax.set_yticks([y + 0.5 for y in range(ROWS)])
        ax.set_xticklabels([f"{c+1}" for c in range(COLS)])
        ax.set_yticklabels([f"{r+1}" for r in range(ROWS)])
        
        # 添加网格线
        ax.grid(True, which='both', color='black', linestyle='-', linewidth=1)
        
        # 设置标题
        ax.set_title(f'横向三连方块 - 行: {i+1}, 起始列: {j+1}')
        
        # 保存图片
        filename = f'{output_dir}/pattern_horizontal_r{i+1:02d}_c{j+1:02d}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        generated_files.append(filename)
        total_count += 1

# 生成所有可能的纵向三连方块
print("正在生成纵向三连方块图案...")
for i in tqdm.tqdm(range(ROWS - 2)):
    for j in range(COLS):
        # 创建空白棋盘
        grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        # 标记三个连续方块为灰色
        for k in range(3):
            grid[i + k][j] = 1
        
        # 绘制棋盘
        fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
        ax.imshow(grid, cmap=cmap, interpolation='nearest')
        
        # 调整坐标系统，使方格居中
        ax.set_xticks([x + 0.5 for x in range(COLS)])
        ax.set_yticks([y + 0.5 for y in range(ROWS)])
        ax.set_xticklabels([f"{c+1}" for c in range(COLS)])
        ax.set_yticklabels([f"{r+1}" for r in range(ROWS)])
        
        # 添加网格线
        ax.grid(True, which='both', color='black', linestyle='-', linewidth=1)
        
        # 设置标题
        ax.set_title(f'纵向三连方块 - 起始行: {i+1}, 列: {j+1}')
        
        # 保存图片
        filename = f'{output_dir}/pattern_vertical_r{i+1:02d}_c{j+1:02d}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        generated_files.append(filename)
        total_count += 1

# 生成汇总文件
with open(f'{output_dir}/patterns_summary.txt', 'w', encoding='utf-8') as f:
    f.write(f'总共生成了 {total_count} 种三格矩形图案\n')
    f.write('=' * 50 + '\n')
    for idx, file in enumerate(generated_files, 1):
        f.write(f"{idx}. {os.path.basename(file)}\n")

print(f"已成功生成所有 {total_count} 种三格矩形图案")
print(f"图片已保存到 '{output_dir}' 目录中")
print(f"图案汇总信息已保存到 '{output_dir}/patterns_summary.txt'")