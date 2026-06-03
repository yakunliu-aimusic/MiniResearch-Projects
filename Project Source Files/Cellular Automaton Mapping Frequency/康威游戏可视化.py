import numpy as np
import os
import matplotlib.pyplot as plt

# 核心配置：固定3×3网格，确保无偏差
GRID_SIZE = 3  # 3 rows × 3 columns
CELL_ON = 1    # Alive (black)
CELL_OFF = 0   # Dead (white)
TOTAL_ITERATIONS = 5  # 5 iterations (6 generations total: 0~5)
OUTPUT_DIR = "3x3_Conway_Evolution"  # Image output folder

def initialize_grid():
    """Manually set 3×3 initial grid (classic demo layout for rule verification)"""
    # Initial state matrix: [[0,1,0],[0,0,1],[1,1,1]]
    return np.array([
        [CELL_OFF, CELL_ON, CELL_OFF],
        [CELL_OFF, CELL_OFF, CELL_ON],
        [CELL_ON, CELL_ON, CELL_ON]
    ], dtype=int)

def count_neighbors(grid, row, col):
    """Count number of alive neighbors (8 Moore neighbors, non-toroidal boundary)"""
    neighbor_count = 0
    # Iterate over 8 neighbor directions
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue  # Skip the cell itself
            # Calculate neighbor coordinates
            nr, nc = row + dr, col + dc
            # Non-toroidal boundary: neighbors outside 3×3 grid are considered dead
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                neighbor_count += grid[nr, nc]
    return neighbor_count

def apply_conway_rules(current_grid):
    """Apply Conway's Game of Life rules to generate next generation grid"""
    next_grid = np.zeros_like(current_grid)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            current_state = current_grid[row, col]
            neighbors = count_neighbors(current_grid, row, col)
            
            # Rule 1: Alive cell with 2-3 neighbors → remains alive
            if current_state == CELL_ON:
                next_grid[row, col] = CELL_ON if neighbors in (2, 3) else CELL_OFF
            # Rule 4: Dead cell with exactly 3 neighbors → becomes alive
            else:
                next_grid[row, col] = CELL_ON if neighbors == 3 else CELL_OFF
    return next_grid

def export_grid_image(grid, generation):
    """Export current 3×3 grid as image (standard table format)"""
    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Set image size and layout (adapt to 3×3 grid without stretching)
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot 3×3 grid: black=alive, white=dead, with grid lines
    im = ax.imshow(grid, cmap="binary", interpolation="nearest", aspect="equal")
    
    # Add bold grid lines to separate cells clearly
    ax.grid(True, color="gray", linewidth=2, alpha=0.8)
    # Force display 0~2 ticks (corresponding to 3 rows/columns)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    # Label axes with English (avoid invalid characters)
    ax.set_xlabel("Column Index", fontsize=12, labelpad=10)
    ax.set_ylabel("Row Index", fontsize=12, labelpad=10)
    ax.tick_params(labelsize=10)
    
    # Title with current generation
    ax.set_title(f"3×3 Conway's Game of Life - Generation {generation}", fontsize=14, pad=15)
    
    # Save image (high resolution, no cropping)
    img_path = os.path.join(OUTPUT_DIR, f"3x3_Gen_{generation:02d}.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return img_path

def main():
    """Main process: Initialize → Iterate → Export images"""
    # 1. Initialize 3×3 grid
    current_grid = initialize_grid()
    print(f"📌 Initial Grid (Generation 0):")
    print(current_grid)
    print("-" * 30)
    
    # 2. Export initial generation (Generation 0) image
    init_img_path = export_grid_image(current_grid, generation=0)
    print(f"✅ Generation 0 image saved: {init_img_path}")
    
    # 3. Execute 5 iterations, export image each time
    for iter_num in range(1, TOTAL_ITERATIONS + 1):
        current_grid = apply_conway_rules(current_grid)
        img_path = export_grid_image(current_grid, generation=iter_num)
        print(f"✅ Generation {iter_num} image saved: {img_path}")
        print(f"   Generation {iter_num} grid state:")
        print(current_grid)
        print("-" * 30)
    
    print(f"\n🎉 All tasks completed!")
    print(f"📁 Total {1 + TOTAL_ITERATIONS} 3×3 grid images saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    # Check required libraries
    required_libs = ["numpy", "matplotlib"]
    missing_libs = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)
    
    if missing_libs:
        print(f"⚠️ Missing dependencies. Please install first:")
        print(f"pip install {' '.join(missing_libs)}")
        exit(1)
    
    # Run main program
    main()