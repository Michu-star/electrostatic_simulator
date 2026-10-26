"""Standalone convergence benchmark comparing Jacobi, Red-Black Gauss-Seidel (w=1),
and Red-Black SOR (w_opt) using the existing ElectrostaticSystem and residual functions."""

import numpy as np
import matplotlib.pyplot as plt

from physics import ElectrostaticSystem, residual, optimal_omega, set_ratios, update_edges


def track_jacobi(system, tol=1e-6, epsilon_0=8.85e-12, max_iter=15_000, check_every=50):
    grid = system.grid.copy()
    free = ~system.fixed
    coeff_x, coeff_y, source_ratio = set_ratios(system)

    history = [(0, residual(system, grid, epsilon_0))]

    for it in range(1, max_iter + 1):
        old_grid = grid.copy()
        new_grid = grid.copy()

        new_grid[1:-1, 1:-1] = (
            coeff_x * (old_grid[1:-1, :-2] + old_grid[1:-1, 2:])
            + coeff_y * (old_grid[2:, 1:-1] + old_grid[:-2, 1:-1])
            + source_ratio * system.rho[1:-1, 1:-1] / epsilon_0
        )
        update_edges(new_grid, system.fixed)
        grid[free] = new_grid[free]

        if it % check_every == 0 or it == max_iter:
            res = residual(system, grid, epsilon_0)
            history.append((it, res))
            if res < tol:
                break

    return np.array(history)


def track_rb_sor(system, w=None, tol=1e-6, epsilon_0=8.85e-12, max_iter=15_000, check_every=50):
    grid = system.grid.copy()
    fixed = system.fixed
    rho = system.rho

    r, c = np.indices(grid.shape)
    free_interior = ~fixed[1:-1, 1:-1]
    red_interior = ((r[1:-1, 1:-1] + c[1:-1, 1:-1]) % 2 == 0) & free_interior
    black_interior = ((r[1:-1, 1:-1] + c[1:-1, 1:-1]) % 2 == 1) & free_interior

    coeff_x, coeff_y, source_ratio = set_ratios(system)
    if w is None:
        w = optimal_omega(system)

    history = [(0, residual(system, grid, epsilon_0))]

    for it in range(1, max_iter + 1):
        for mask in (red_interior, black_interior):
            stencil = (
                coeff_y * (grid[:-2, 1:-1] + grid[2:, 1:-1])
                + coeff_x * (grid[1:-1, :-2] + grid[1:-1, 2:])
                + source_ratio * rho[1:-1, 1:-1] / epsilon_0
            )
            grid[1:-1, 1:-1][mask] = (1 - w) * grid[1:-1, 1:-1][mask] + w * stencil[mask]

        update_edges(grid, fixed)

        if it % check_every == 0 or it == max_iter:
            res = residual(system, grid, epsilon_0)
            history.append((it, res))
            if res < tol:
                break

    return np.array(history)


def main():
    plane = ElectrostaticSystem((201, 201), 5, 10)
    plane.add_square_electrode(0.5, 1, 4.5, 1 + plane.dy, 200)
    plane.add_square_electrode(0.5, 9, 4.5, 9 + plane.dy, -200)
    plane.add_point_charge(4, 6, 1e-9)

    plane.set_potential((0, slice(None)), 0)
    plane.set_potential((-1, slice(None)), 0)
    plane.set_potential((slice(None), 0), 0)
    plane.set_potential((slice(None), -1), 0)

    tol = 1e-6
    w_opt = optimal_omega(plane)

    hist_sor = track_rb_sor(plane, w=w_opt, tol=tol)
    hist_gs = track_rb_sor(plane, w=1.0, tol=tol)
    hist_jac = track_jacobi(plane, tol=tol)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(hist_sor[:, 0], hist_sor[:, 1], label=f"Red-Black SOR (ω_opt = {w_opt:.3f})", linewidth=2)
    ax.semilogy(hist_gs[:, 0], hist_gs[:, 1], label="Red-Black Gauss-Seidel (ω = 1.0)", linewidth=2)
    ax.semilogy(hist_jac[:, 0], hist_jac[:, 1], label="Jacobi", linewidth=2)
    ax.axhline(tol, color="grey", linestyle="--", label=f"tol = {tol:.0e}")

    ax.set_title("2D Poisson Solver Convergence (201 × 201 Grid)")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("RMS Poisson Residual")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
