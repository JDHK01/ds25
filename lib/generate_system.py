# 创建了一个字典, 键是点的名称, 值是点的坐标
def generate_coordinate_system():
    """动态生成坐标系统"""
    # 返回字典
    coordinates = {}
    for row in range(1, 8):  # B1 到 B7
        for col in range(1, 10):  # A1 到 A9
            # 从 A1 到 A9 对应 x 坐标从 4.0 到 0.0 (递减)
            x = (9 - col) * 0.5
            # 从 B1 到 B7 对应 y 坐标从 0.0 到 3.0
            y = (row - 1) * 0.5
            point_name = f"A{col}B{row}"
            coordinates[point_name] = (x, y)
    return coordinates