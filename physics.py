import numpy as np


class ElectrostaticGrid:
    def __init__(self, shape, width, height):
        self.grid = np.zeros(shape)
        self.fixed = np.zeros(shape, dtype=bool)
        self.width, self.height = width, height

        self.x = np.linspace(0, self.width, shape[1])
        self.y = np.linspace(0, self.height, shape[0])

        self.dx = width / (shape[1] - 1)
        self.dy = height / (shape[0] - 1)

    def set_potential(self, region, potential):
        self.grid[region] = potential
        self.fixed[region] = True


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
def jacobi(system, tol=1e-6):
    grid = system.grid.copy()
    free = ~system.fixed

    iterations = 0

    while True:
        iterations += 1
        old_grid = grid.copy()
        new_grid = grid.copy()

        # Handle center
        left_grid = old_grid[1:-1, :-2]
        right_grid = old_grid[1:-1, 2:]
        bottom_grid = old_grid[2:, 1:-1]
        top_grid = old_grid[:-2, 1:-1]

        new_grid[1:-1, 1:-1] = 0.25 * (left_grid + right_grid + bottom_grid + top_grid)

        # Handle edges
        new_grid[:, 0] = old_grid[:, 1] # left
        new_grid[:, -1] = old_grid[:, -2] # right
        new_grid[0, :] = old_grid[1, :] # bottom
        new_grid[-1, :] = old_grid[-2, :] # top

        grid[free] = new_grid[free]

        change = np.max(np.abs(grid - old_grid))

        if change < tol:
            break

    print(f'Jacobi finished in {iterations} iterations')

    return grid


# Gauss-Seidel
def gs(system, tol=1e-6):
    grid = system.grid.copy()
    fixed = system.fixed
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
def sor(system, tol=1e-6, w=1.5):
    grid = system.grid.copy()
    fixed = system.fixed
    iterations = 0

    while True:
        iterations += 1
        old_grid = grid.copy()

        # Handle center
        for j in range(1, grid.shape[0] - 1):
            for k in range(1, grid.shape[1] - 1):
                if fixed[j,k]:
                    continue
                else:
                    gs_value = 0.25 * (
                            grid[j + 1, k] + grid[j - 1, k]
                            + grid[j, k + 1] + grid[j, k - 1]
                    )

                    grid[j, k] = (1 - w) * grid[j, k] + w * gs_value

        update_edges(grid, fixed)

        change = np.max(np.abs(grid - old_grid))

        if change < tol:
            break

    print(f'Successive Over-Relaxation finished in {iterations} iterations')

    return grid