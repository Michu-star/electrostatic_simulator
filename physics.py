import numpy as np


class ElectrostaticSystem:
    def __init__(self, shape, width, height):
        self.grid = np.zeros(shape, dtype=float)
        self.fixed = np.zeros(shape, dtype=bool)
        self.rho = np.zeros(shape, dtype=float)
        self.width, self.height = width, height

        self.x = np.linspace(0, self.width, shape[1])
        self.y = np.linspace(0, self.height, shape[0])

        self.dx = width / (shape[1] - 1)
        self.dy = height / (shape[0] - 1)

    def set_potential(self, region, potential):
        self.grid[region] = potential
        self.fixed[region] = True

    def set_charge_density(self, region, rho):
        self.rho[region] = rho


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


def set_ratios(system):
    denominator = 2 * (system.dx ** 2 + system.dy ** 2)

    coeff_x = system.dy ** 2 / denominator
    coeff_y = system.dx ** 2 / denominator
    source_ratio = (system.dx * system.dy) ** 2 / denominator

    return coeff_x, coeff_y, source_ratio


def residual(system, grid, epsilon_0=8.85e-12):
    center = grid[1:-1, 1:-1]

    left = grid[1:-1, :-2]
    right = grid[1:-1, 2:]

    bottom = grid[:-2, 1:-1]
    top = grid[2:, 1:-1]

    laplacian = (
        (left - 2 * center + right) / system.dx**2
        + (bottom - 2 * center + top) / system.dy**2
    )

    poisson_residual = (
        laplacian
        + system.rho[1:-1, 1:-1] / epsilon_0
    )

    free = ~system.fixed[1:-1, 1:-1]

    return np.max(np.abs(poisson_residual[free]))


# Jacobi
def jacobi(system, tol=1e-6, epsilon_0=8.85e-12, max_iter=100_000):
    grid = system.grid.copy()
    free = ~system.fixed

    coeff_x, coeff_y, source_ratio = set_ratios(system)

    iterations = 0

    while iterations < max_iter:
        iterations += 1
        old_grid = grid.copy()
        new_grid = grid.copy()

        # Handle center
        left_grid = old_grid[1:-1, :-2]
        right_grid = old_grid[1:-1, 2:]
        bottom_grid = old_grid[2:, 1:-1]
        top_grid = old_grid[:-2, 1:-1]

        new_grid[1:-1, 1:-1] = (
                coeff_x * (left_grid + right_grid)
                + coeff_y * (bottom_grid + top_grid)
                + source_ratio * system.rho[1:-1, 1:-1] / epsilon_0
        )

        # Handle edges
        new_grid[:, 0] = old_grid[:, 1] # left
        new_grid[:, -1] = old_grid[:, -2] # right
        new_grid[0, :] = old_grid[1, :] # bottom
        new_grid[-1, :] = old_grid[-2, :] # top

        grid[free] = new_grid[free]

        # change = np.max(np.abs(grid - old_grid))

        if iterations % 100 == 0:
            if residual(system, grid, epsilon_0) < tol:
                break

    print(f'Jacobi finished in {iterations} iterations')

    return grid


def checkerboard_mask(shape):
    r, c = np.indices(shape)
    return (r + c) % 2 == 0


# Red-Black Gauss-Seidel
def rb_gs(system, tol=1e-6, epsilon_0=8.85e-12, max_iter=100_000):
    grid = system.grid.copy()
    fixed = system.fixed
    rho = system.rho

    # Precompute interior checkerboard masks
    r, c = np.indices(grid.shape)
    free_interior = ~fixed[1:-1, 1:-1]
    red_interior = ((r[1:-1, 1:-1] + c[1:-1, 1:-1]) % 2 == 0) & free_interior
    black_interior = ((r[1:-1, 1:-1] + c[1:-1, 1:-1]) % 2 == 1) & free_interior

    coeff_x, coeff_y, source_ratio = set_ratios(system)
    iterations = 0

    while iterations < max_iter:
        iterations += 1

        # 1. Update red cells
        stencil = (
                coeff_y * (grid[:-2, 1:-1] + grid[2:, 1:-1])
                + coeff_x * (grid[1:-1, :-2] + grid[1:-1, 2:])
                + source_ratio * rho[1:-1, 1:-1] / epsilon_0
        )
        grid[1:-1, 1:-1][red_interior] = stencil[red_interior]

        # 2. Update black cells
        stencil = (
                coeff_y * (grid[:-2, 1:-1] + grid[2:, 1:-1])
                + coeff_x * (grid[1:-1, :-2] + grid[1:-1, 2:])
                + source_ratio * rho[1:-1, 1:-1] / epsilon_0
        )
        grid[1:-1, 1:-1][black_interior] = stencil[black_interior]

        # 3. Handle boundary conditions
        update_edges(grid, fixed)

        # Check residual periodically
        if iterations % 50 == 0:
            if residual(system, grid, epsilon_0) < tol:
                break

    print(f'Red-Black Gauss-Seidel finished in {iterations} iterations')

    return grid


# Gauss-Seidel
def gs(system, tol=1e-6, epsilon_0=8.85e-12, max_iter=10_000):
    grid = system.grid.copy()
    fixed = system.fixed

    coeff_x, coeff_y, source_ratio = set_ratios(system)

    iterations = 0

    while iterations < max_iter:
        iterations += 1
        # old_grid = grid.copy()

        # Handle center
        for j in range(1, grid.shape[0] - 1):
            for k in range(1, grid.shape[1] - 1):
                if fixed[j, k]:
                    continue
                else:
                    grid[j,k] = (
                            coeff_y * (grid[j + 1, k] + grid[j - 1, k])
                            + coeff_x * (grid[j, k + 1] + grid[j, k - 1])
                            + source_ratio * system.rho[j, k] / epsilon_0
                    )

        update_edges(grid, fixed)

        # change = np.max(np.abs(grid - old_grid))

        if iterations % 50 == 0:
            if residual(system, grid, epsilon_0) < tol:
                break

    print(f'Gauss-Seidel finished in {iterations} iterations')

    return grid


# Successive Over-Relaxation
def sor(system, tol=1e-6, w=1.5, epsilon_0=8.85e-12, max_iter=10_000):
    grid = system.grid.copy()
    fixed = system.fixed

    coeff_x, coeff_y, source_ratio = set_ratios(system)

    iterations = 0

    while iterations < max_iter:
        iterations += 1
        # old_grid = grid.copy()

        # Handle center
        for j in range(1, grid.shape[0] - 1):
            for k in range(1, grid.shape[1] - 1):
                if fixed[j,k]:
                    continue
                else:
                    gs_value = (
                            coeff_y * (grid[j + 1, k] + grid[j - 1, k])
                            + coeff_x * (grid[j, k + 1] + grid[j, k - 1])
                            + source_ratio * system.rho[j, k] / epsilon_0
                    )

                    grid[j, k] = (1 - w) * grid[j, k] + w * gs_value

        update_edges(grid, fixed)

        # change = np.max(np.abs(grid - old_grid))

        if iterations % 50 == 0:
            if residual(system, grid, epsilon_0) < tol:
                break

    print(f'Successive Over-Relaxation finished in {iterations} iterations')

    return grid