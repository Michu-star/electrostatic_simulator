import numpy as np
import matplotlib.pyplot as plt

from physics import jacobi, gs, sor
from plotting import draw_grid

# Set boundaries
initial_grid = np.zeros((100,100))
initial_grid[49:51, 10] = 10.0
initial_grid[49:51, -10] = -10.0

fixed_points = np.zeros_like(initial_grid, dtype=bool)
fixed_points[49:51, 10] = True
fixed_points[49:51, -10] = True

V = jacobi(initial_grid, fixed_points)
dV_dy, dV_dx = np.gradient(V)

Ex = -dV_dx
Ey = -dV_dy

y, x = np.mgrid[0:V.shape[0], 0:V.shape[1]]


fig, (ax_V, ax_E) = plt.subplots(1, 2)

ax_V.set_title("V")
draw_grid(ax_V, V)

ax_E.set_title("E")
ax_E.streamplot(x, y, Ex, Ey)
ax_E.set_aspect("equal")

plt.show()