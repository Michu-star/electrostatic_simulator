import numpy as np
import matplotlib.pyplot as plt


# Set boundaries
initial_grid = np.zeros((10,10))
initial_grid[:, 0] = 1.0
initial_grid[:, -1] = 1.0
initial_grid[0] = 1.0
initial_grid[-1] = 1.0

# Gauss-Seidel
def gs(initial_grid):
    grid = initial_grid.copy()
    iterations = 0

    while True:
        iterations += 1
        old_grid = grid.copy()

        for j in range(1, grid.shape[0] - 1):
            for k in range(1, grid.shape[1] - 1):
                grid[j,k] = 0.25 * (
                    grid[j + 1, k] + grid[j - 1, k]
                    + grid[j, k + 1] + grid[j, k - 1]
                )

        change = np.max(np.abs(grid - old_grid))

        if change < 1e-6:
            break

    print(f'Gauss-Seidel finished in {iterations} iterations')

    return grid

# Successive Over-Relaxation
def sor(initial_grid, w):
    grid = initial_grid.copy()
    iterations = 0

    while True:
        iterations += 1
        old_grid = grid.copy()
        gs_grid = grid.copy()

        for j in range(1, grid.shape[0] - 1):
            for k in range(1, grid.shape[1] - 1):
                gs_grid[j,k] = 0.25 * (
                    gs_grid[j + 1, k] + gs_grid[j - 1, k]
                    + gs_grid[j, k + 1] + gs_grid[j, k - 1]
                )

        grid = (1 - w) * old_grid + w * gs_grid

        change = np.max(np.abs(grid - old_grid))

        if change < 1e-6:
            break

    print(f'Successive Over-Relaxation finished in {iterations} iterations')

    return grid


def draw_grid(ax, grid):
    image = ax.imshow(grid, origin="lower")
    ax.figure.colorbar(image, ax=ax, label="Potential V")


fig, (ax_gs, ax_sor) = plt.subplots(1, 2)

ax_gs.set_title("Gauss-Seidel Grid Plot")
draw_grid(ax_gs, gs(initial_grid))

ax_sor.set_title("Successive Over-Relaxation Grid Plot")
draw_grid(ax_sor, sor(initial_grid, 1.5))

plt.show()