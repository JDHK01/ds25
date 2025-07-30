import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import copy

class GridPathPlanner:
    def __init__(self, width=9, height=7):
        self.width = width  # A1-A9
        self.height = height  # B1-B7
        self.grid = np.zeros((height, width))
        self.no_fly_zones = set()
        self.start_pos = (0, 8)  # A9B1 -> (row=0, col=8)
        self.visited = set()
        self.path = []
        
    def add_no_fly_zone(self, col_letter, row_num):
        """Add a no-fly zone at specified position (e.g., 'A', 1)"""
        col = ord(col_letter.upper()) - ord('A')
        row = row_num - 1
        if 0 <= col < self.width and 0 <= row < self.height:
            self.no_fly_zones.add((row, col))
            self.grid[row, col] = -1
    
    def add_consecutive_no_fly_zones(self, positions):
        """Add multiple consecutive no-fly zones
        Args:
            positions: List of (col_letter, row_num) tuples
        """
        for col_letter, row_num in positions:
            self.add_no_fly_zone(col_letter, row_num)
    
    def create_example_consecutive_zones(self, pattern='horizontal'):
        """Create example patterns of 3 consecutive no-fly zones"""
        self.no_fly_zones.clear()
        self.grid = np.zeros((self.height, self.width))
        
        if pattern == 'horizontal':
            # Horizontal line of 3 zones in middle
            self.add_consecutive_no_fly_zones([('D', 4), ('E', 4), ('F', 4)])
        elif pattern == 'vertical':
            # Vertical line of 3 zones
            self.add_consecutive_no_fly_zones([('E', 3), ('E', 4), ('E', 5)])
        elif pattern == 'L_shape':
            # L-shaped pattern
            self.add_consecutive_no_fly_zones([('D', 4), ('E', 4), ('E', 5)])
        elif pattern == 'diagonal':
            # Diagonal pattern
            self.add_consecutive_no_fly_zones([('C', 3), ('D', 4), ('E', 5)])
        else:
            # Default: corner block
            self.add_consecutive_no_fly_zones([('C', 2), ('C', 3), ('D', 3)])
    
    def is_valid_move(self, row, col):
        """Check if move is valid (within bounds and not a no-fly zone)"""
        return (0 <= row < self.height and 
                0 <= col < self.width and 
                (row, col) not in self.no_fly_zones)
    
    def get_neighbors(self, row, col):
        """Get valid neighboring positions"""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_move(new_row, new_col):
                neighbors.append((new_row, new_col))
        return neighbors
    
    def find_path_dfs(self, current_pos=None, path=None, visited=None):
        """Find path using DFS to minimize repeated visits"""
        if current_pos is None:
            current_pos = self.start_pos
        if path is None:
            path = [current_pos]
        if visited is None:
            visited = {current_pos}
        
        # If we have visited most cells and can return to start
        if len(visited) >= self.width * self.height - len(self.no_fly_zones) - 5:
            neighbors = self.get_neighbors(current_pos[0], current_pos[1])
            if self.start_pos in neighbors:
                return path + [self.start_pos]
        
        # Try to visit unvisited neighbors first
        neighbors = self.get_neighbors(current_pos[0], current_pos[1])
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        
        # Sort by distance to unvisited areas to encourage exploration
        if unvisited_neighbors:
            neighbors_to_try = unvisited_neighbors
        else:
            neighbors_to_try = neighbors
        
        for next_pos in neighbors_to_try:
            new_visited = visited.copy()
            new_visited.add(next_pos)
            new_path = path + [next_pos]
            
            # If we can return to start and have a good coverage
            if (len(new_visited) >= max(20, self.width * self.height - len(self.no_fly_zones) - 10)):
                if self.start_pos in self.get_neighbors(next_pos[0], next_pos[1]):
                    return new_path + [self.start_pos]
            
            # Continue exploring if path not too long
            if len(new_path) < 100:
                result = self.find_path_dfs(next_pos, new_path, new_visited)
                if result:
                    return result
        
        return None
    
    def find_spiral_path(self):
        """Find a spiral-like path that covers most cells"""
        path = [self.start_pos]
        visited = {self.start_pos}
        current = self.start_pos
        
        # Directions: right, up, left, down (clockwise spiral)
        directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        dir_idx = 0
        
        while len(visited) < self.width * self.height - len(self.no_fly_zones):
            moved = False
            
            # Try current direction
            dr, dc = directions[dir_idx]
            new_row, new_col = current[0] + dr, current[1] + dc
            
            if (self.is_valid_move(new_row, new_col) and 
                (new_row, new_col) not in visited):
                current = (new_row, new_col)
                path.append(current)
                visited.add(current)
                moved = True
            else:
                # Change direction
                dir_idx = (dir_idx + 1) % 4
                
                # Try new direction
                dr, dc = directions[dir_idx]
                new_row, new_col = current[0] + dr, current[1] + dc
                
                if (self.is_valid_move(new_row, new_col) and 
                    (new_row, new_col) not in visited):
                    current = (new_row, new_col)
                    path.append(current)
                    visited.add(current)
                    moved = True
            
            if not moved:
                # Find nearest unvisited cell
                unvisited = []
                for r in range(self.height):
                    for c in range(self.width):
                        if (r, c) not in visited and (r, c) not in self.no_fly_zones:
                            unvisited.append((r, c))
                
                if not unvisited:
                    break
                
                # Move to closest unvisited cell
                closest = min(unvisited, 
                            key=lambda x: abs(x[0] - current[0]) + abs(x[1] - current[1]))
                current = closest
                path.append(current)
                visited.add(current)
        
        # Return to start
        if current != self.start_pos:
            path.append(self.start_pos)
        
        return path
    
    def visualize_path(self, path=None, title="Path Planning Visualization"):
        """Visualize the grid and path"""
        if path is None:
            path = self.path
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Draw grid
        for i in range(self.height + 1):
            ax.axhline(y=i, color='lightgray', linewidth=0.5)
        for i in range(self.width + 1):
            ax.axvline(x=i, color='lightgray', linewidth=0.5)
        
        # Draw no-fly zones
        for row, col in self.no_fly_zones:
            rect = patches.Rectangle((col, self.height - 1 - row), 1, 1, 
                                   linewidth=2, edgecolor='red', facecolor='red', alpha=0.7)
            ax.add_patch(rect)
            ax.text(col + 0.5, self.height - 1 - row + 0.5, 'X', 
                   ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        
        # Draw path
        if path and len(path) > 1:
            path_x = [col + 0.5 for row, col in path]
            path_y = [self.height - 1 - row + 0.5 for row, col in path]
            
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Path')
            
            # Mark waypoints
            for i, (x, y) in enumerate(zip(path_x, path_y)):
                if i == 0:
                    ax.plot(x, y, 'go', markersize=10, label='Start/End')
                elif i == len(path_x) - 1:
                    ax.plot(x, y, 'go', markersize=10)
                else:
                    ax.plot(x, y, 'bo', markersize=4)
                
                # Add step numbers for first few and last few points
                if i < 5 or i >= len(path_x) - 3:
                    ax.text(x + 0.1, y + 0.1, str(i), fontsize=8, color='darkblue')
        
        # Set labels
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(0.5, self.width + 0.5))
        ax.set_xticklabels([f'A{i+1}' for i in range(self.width)])
        ax.set_yticks(np.arange(0.5, self.height + 0.5))
        ax.set_yticklabels([f'B{self.height-i}' for i in range(self.height)])
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig, ax
    
    def print_path_info(self, path):
        """Print path information"""
        if not path:
            print("No path found!")
            return
        
        print(f"Path length: {len(path)} steps")
        print(f"Unique cells visited: {len(set(path))}")
        print(f"Total cells in grid: {self.width * self.height}")
        print(f"No-fly zones: {len(self.no_fly_zones)}")
        print(f"Coverage: {len(set(path)) / (self.width * self.height - len(self.no_fly_zones)) * 100:.1f}%")
        
        # Convert path to readable format
        readable_path = []
        for row, col in path[:10]:  # Show first 10 steps
            col_letter = chr(ord('A') + col)
            row_num = row + 1
            readable_path.append(f"{col_letter}{row_num}")
        
        print(f"First 10 steps: {' -> '.join(readable_path)}")
        
        if len(path) > 10:
            readable_path = []
            for row, col in path[-5:]:  # Show last 5 steps
                col_letter = chr(ord('A') + col)
                row_num = row + 1
                readable_path.append(f"{col_letter}{row_num}")
            print(f"Last 5 steps: {' -> '.join(readable_path)}")

def main():
    """Demo different no-fly zone patterns and path planning algorithms"""
    patterns = ['horizontal', 'vertical', 'L_shape', 'diagonal', 'corner']
    
    for pattern in patterns:
        print(f"\n{'='*50}")
        print(f"TESTING PATTERN: {pattern.upper()}")
        print('='*50)
        
        # Create path planner
        planner = GridPathPlanner()
        
        # Set up consecutive no-fly zones
        planner.create_example_consecutive_zones(pattern)
        
        print(f"Start position: A9B1")
        print(f"No-fly zones: {[(chr(ord('A') + col), row + 1) for row, col in planner.no_fly_zones]}")
        
        # Find path using spiral method (usually gives better coverage)
        print(f"\nFinding spiral path for {pattern} pattern...")
        spiral_path = planner.find_spiral_path()
        
        print(f"\n=== Spiral Path Results for {pattern} ===")
        planner.print_path_info(spiral_path)
        
        # Visualize
        fig1, ax1 = planner.visualize_path(spiral_path, f"Spiral Path - {pattern.title()} No-fly Zones")
        plt.show()
        
        # Also try DFS method for comparison
        print(f"\nFinding DFS path for {pattern} pattern...")
        dfs_path = planner.find_path_dfs()
        
        if dfs_path:
            print(f"\n=== DFS Path Results for {pattern} ===")
            planner.print_path_info(dfs_path)
            
            fig2, ax2 = planner.visualize_path(dfs_path, f"DFS Path - {pattern.title()} No-fly Zones")
            plt.show()
        else:
            print(f"DFS could not find a suitable path for {pattern} pattern")
        
        input(f"\nPress Enter to continue to next pattern...")

def demo_single_pattern(pattern='horizontal'):
    """Run demo for a single pattern - useful for quick testing"""
    print(f"Testing {pattern} pattern of consecutive no-fly zones")
    print("-" * 50)
    
    planner = GridPathPlanner()
    planner.create_example_consecutive_zones(pattern)
    
    print(f"Grid: 9x7 (A1-A9, B1-B7)")
    print(f"Start/End: A9B1")
    print(f"No-fly zones ({pattern}): {[(chr(ord('A') + col), row + 1) for row, col in planner.no_fly_zones]}")
    
    # Test spiral algorithm
    spiral_path = planner.find_spiral_path()
    print(f"\n=== Spiral Algorithm Results ===")
    planner.print_path_info(spiral_path)
    
    # Visualize
    fig, ax = planner.visualize_path(spiral_path, f"Path Planning: {pattern.title()} No-fly Zones")
    plt.show()
    
    return planner, spiral_path

if __name__ == "__main__":
    # Uncomment one of the following options:
    
    # Option 1: Test all patterns (full demo)
    # main()
    
    # Option 2: Test single pattern (quick demo)
    demo_single_pattern('L_shape')  # Try: horizontal, vertical, L_shape, diagonal, corner
    
    # Option 3: Custom no-fly zones
    # planner = GridPathPlanner()
    # planner.add_consecutive_no_fly_zones([('D', 3), ('D', 4), ('E', 4)])  # Custom L-shape
    # path = planner.find_spiral_path()
    # planner.print_path_info(path)
    # planner.visualize_path(path, "Custom No-fly Zones")
    # plt.show()