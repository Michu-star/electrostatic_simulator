import numpy as np
import matplotlib.pyplot as plt

from physics import ElectrostaticSystem, jacobi, gs, sor
from plotting import draw_grid, draw_streamplot


def main():
    plane = ElectrostaticSystem((101, 101), 5, 10)

    # Positive electrode
    plane.set_potential((50, 10), 10.0)

    # Negative electrode
    # plane.set_potential((50, -10), -10.0)

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