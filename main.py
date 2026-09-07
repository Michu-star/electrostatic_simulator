import numpy as np
import matplotlib.pyplot as plt

# Set boundaries
initial_grid = np.zeros((10,10))
initial_grid[:, 0] = 1.0

fixed_points = np.zeros_like(initial_grid, dtype=bool)
fixed_points[:, 0] = True
fixed_points[3, 3] = True


def update_edges(grid, fixed):
    for i in range(0, grid.shape[0]):
        # left edge
        if not fixed[i, 0]:
            grid[i, 0] = grid[i, 1]

        # right edge
        if not fixed[i, -1]:
            grid[i, -1] = grid[i, -2]

    for j in range(0, grid.shape[1]):
        # bottom edge
        if not fixed[0, j]:
            grid[0, j] = grid[1, j]

        # top edge
        if not fixed[-1, j]:
            grid[-1, j] = grid[-2, j]


# Jacobi
def jacobi(initial_grid, fixed, tol=1e-6):
    grid = initial_grid.copy()
    free = ~fixed

    iterations = 0

    while True:
        iterations += 1
        old_grid = grid.copy()
        new_grid = grid.copy()

        # Handle center
        left_grid = new_grid[1:-1, :-2]
        right_grid = new_grid[1:-1, 2:]
        bottom_grid = new_grid[2:, 1:-1]
        top_grid = new_grid[:-2, 1:-1]

        new_grid[1:-1, 1:-1] = 0.25 * (left_grid + right_grid + bottom_grid + top_grid)

        # Handle edges
        new_grid[0, :] = old_grid[1, :]
        new_grid[-1, :] = old_grid[-2, :]
        new_grid[:, 0] = old_grid[:, 1]
        new_grid[:, -1] = old_grid[:, -2]

        grid[free] = new_grid[free]

        change = np.max(np.abs(grid - old_grid))

        if change < tol:
            break

    print(f'Jacobi finished in {iterations} iterations')

    return grid


# Gauss-Seidel
def gs(initial_grid, fixed, tol=1e-6):
    grid = initial_grid.copy()
    iterations = 0

    while True:
        iterations += 1
        old_grid = grid.copy()

        # Handle center
        for j in range(1, grid.shape[0] - 1):
            for k in range(1, grid.shape[1] - 1):
                if fixed[j, k]:
                    continue
                else:
                    grid[j,k] = 0.25 * (
                        grid[j + 1, k] + grid[j - 1, k]
                        + grid[j, k + 1] + grid[j, k - 1]
                    )

        update_edges(grid, fixed)

        change = np.max(np.abs(grid - old_grid))

        if change < tol:
            break

    print(f'Gauss-Seidel finished in {iterations} iterations')

    return grid


# Successive Over-Relaxation
def sor(initial_grid, fixed, tol=1e-6, w=1.5):
    grid = initial_grid.copy()
    iterations = 0

    while True:
        iterations += 1
        old_grid = grid.copy()
        gs_grid = grid.copy()

        # Handle center
        for j in range(1, grid.shape[0] - 1):
            for k in range(1, grid.shape[1] - 1):
                if fixed[j,k]:
                    continue
                else:
                    gs_grid[j,k] = 0.25 * (
                        gs_grid[j + 1, k] + gs_grid[j - 1, k]
                        + gs_grid[j, k + 1] + gs_grid[j, k - 1]
                    )

        grid = (1 - w) * old_grid + w * gs_grid

        update_edges(grid, fixed)

        change = np.max(np.abs(grid - old_grid))

        if change < tol:
            break

    print(f'Successive Over-Relaxation finished in {iterations} iterations')

    return grid


def draw_grid(ax, grid):
    image = ax.imshow(grid, origin="lower")
    ax.figure.colorbar(image, ax=ax, label="Potential V")


fig, (ax_j, ax_gs, ax_sor) = plt.subplots(1, 3)

ax_j.set_title("Jacobi")
draw_grid(ax_j, jacobi(initial_grid, fixed_points))

ax_gs.set_title("Gauss-Seidel")
draw_grid(ax_gs, gs(initial_grid, fixed_points))

ax_sor.set_title("Successive Over-Relaxation")
draw_grid(ax_sor, sor(initial_grid, fixed_points, w=1.5))

plt.show()