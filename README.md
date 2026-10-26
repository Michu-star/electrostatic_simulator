# 2D Electrostatic Poisson Solver (`electrostatic_simulator`)

A 2D finite-difference solver for the electrostatic Poisson and Laplace equations on rectangular grids with arbitrary Dirichlet electrodes, point charges, and free (Neumann) boundary segments, written in Python with `NumPy` and `Matplotlib`.

## Physics & Numerical Discretization

The solver computes the electrostatic potential $V(x, y)$ satisfying the 2D Poisson equation:

$$
\nabla^2 V(x, y) = \frac{\partial^2 V}{\partial x^2} + \frac{\partial^2 V}{\partial y^2} = -\frac{\rho(x, y)}{\varepsilon_0}
$$

and derives the electric field via finite-difference gradients:

$$
\mathbf{E}(x, y) = -\nabla V(x, y) = \left(-\frac{\partial V}{\partial x}, -\frac{\partial V}{\partial y}\right)
$$

### 5-Point Finite-Difference Stencil (Anisotropic Grid $\Delta x, \Delta y$)

On an $N_y \times N_x$ grid, the interior residual at free nodes is defined as:

$$
R_{i,j} = \frac{V_{i,j-1} - 2V_{i,j} + V_{i,j+1}}{\Delta x^2} + \frac{V_{i-1,j} - 2V_{i,j} + V_{i+1,j}}{\Delta y^2} + \frac{\rho_{i,j}}{\varepsilon_0}
$$

Convergence is monitored via the mesh-independent **root-mean-square (RMS) Poisson residual** across all non-fixed interior cells:

$$
\|R\|_{\text{RMS}} = \sqrt{\frac{1}{N_{\text{free}}} \sum_{(i,j)} R_{i,j}^2} < \text{tol}
$$

## Iterative Relaxation Methods & Vectorization

1. **Vectorized Red–Black Successive Over-Relaxation (`rb_sor`)**:
   - Partitions the 2D grid into two decoupled checkerboard sublattices:
     - **Red sites**: $(i + j) \bmod 2 = 0$
     - **Black sites**: $(i + j) \bmod 2 = 1$
   - Because the 5-point Laplacian stencil only couples Red sites to Black neighbors (and vice versa), each half-sweep can be fully vectorized in `NumPy` without Python loops.
   - Uses the analytical **optimal relaxation parameter** $\omega_{\text{opt}}$ derived from the Jacobi spectral radius $\rho_J$:

$$
\rho_J = \frac{\Delta y^2 \cos\left(\frac{\pi}{N_x - 1}\right) + \Delta x^2 \cos\left(\frac{\pi}{N_y - 1}\right)}{\Delta x^2 + \Delta y^2}, \qquad \omega_{\text{opt}} = \frac{2}{1 + \sqrt{1 - \rho_J^2}}
$$

2. **Vectorized Jacobi Iteration (`jacobi`)**:
   - Synchronous full-grid update using the previous step's potential values.

## Project Structure

- `physics.py` — `ElectrostaticSystem` class, boundary condition handling (`update_edges`), RMS Poisson residual (`residual`), optimal relaxation factor (`optimal_omega`), and vectorized solvers (`rb_sor`, `jacobi`).
- `plotting.py` — Potential heatmap (`draw_grid`) and electric field streamplot (`draw_streamplot`) routines.
- `main.py` — Example setup with parallel finite electrodes, a point charge, and grounded boundary edges.

## Installation & Usage

```bash
pip install -r requirements.txt
python main.py
```
