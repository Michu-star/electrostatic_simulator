import numpy as np
import matplotlib.pyplot as plt


# Set boundaries
initial_grid = np.zeros((10,10))
initial_grid[:, 0] = 1.0
initial_grid[:, -1] = 1.0
initial_grid[0] = 1.0
initial_grid[-1] = 1.0


def gauss_seidel(initial_grid):
    grid = initial_grid.copy()
    iterations = 0

    while True:
        iterations += 1
        previous_grid = grid.copy()

        for j in range(1, grid.shape[0] - 1):
            for k in range(1, grid.shape[1] - 1):
                grid[j,k] = 0.25 * (
                    grid[j + 1, k] + grid[j - 1, k]
                    + grid[j, k + 1] + grid[j, k - 1]
                )

        change = np.max(np.abs(grid - previous_grid))

        if change < 1e-6:
            break

    print(f'Gauss-Seidel finished in {iterations} iterations')

    return grid


def draw_grid(ax, grid):
    image = ax.imshow(grid, origin="lower")
    ax.figure.colorbar(image, ax=ax, label="Potential V")


fig, ax = plt.subplots()
draw_grid(ax, gauss_seidel(initial_grid))

plt.show()