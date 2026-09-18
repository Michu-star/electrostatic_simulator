import numpy as np
import matplotlib.pyplot as plt

from physics import jacobi, gs, sor
from plotting import draw_grid


class ElectrostaticGrid:
    def __init__(self, shape):
        self.grid = np.zeros(shape)
        self.fixed = np.zeros(shape, dtype=bool)

    def set_potential(self, region, potential):
        self.grid[region] = potential
        self.fixed[region] = True


initial = ElectrostaticGrid((100, 100))
# Positive point charge
initial.set_potential((slice(49, 51), 10), 10.0)

# Negative point charge
initial.set_potential((slice(49, 51), -10), -10.0)


V = jacobi(initial)
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