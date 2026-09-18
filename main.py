import numpy as np
import matplotlib.pyplot as plt

from physics import jacobi, gs, sor
from plotting import draw_grid, draw_streamplot


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


def main():
    plane = ElectrostaticGrid((101, 101), 10, 10)

    # Positive electrode
    plane.set_potential((50, 10), 10.0)

    # Negative electrode
    plane.set_potential((50, -10), -10.0)

    V = jacobi(plane)
    dV_dy, dV_dx = np.gradient(V, plane.dy, plane.dx)

    Ex = -dV_dx
    Ey = -dV_dy


    fig, (ax_V, ax_E) = plt.subplots(1, 2)

    ax_V.set_title("V")
    draw_grid(ax_V, V, plane)

    ax_E.set_title("E")
    draw_streamplot(ax_E, (Ex, Ey,), plane)

    plt.show()

if __name__ == "__main__":
    main()