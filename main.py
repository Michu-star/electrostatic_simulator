import numpy as np
import matplotlib.pyplot as plt

from physics import ElectrostaticSystem, jacobi, gs, sor, rb_sor
from plotting import draw_grid, draw_streamplot


def main():
    plane = ElectrostaticSystem((101, 101), 5, 10)

    # Positive electrode
    #plane.set_potential((50, 10), 10.0)

    # Negative electrode
    # plane.set_potential((50, -10), -10.0)

    # Make edges zero
    plane.set_potential((0, slice(None)), 0)
    plane.set_potential((-1, slice(None)), 0)
    plane.set_potential((slice(None), 0), 0)
    plane.set_potential((slice(None), -1), 0)

    # Charge density
    plane.set_charge_density((50, 10), 1e-9)
    plane.set_charge_density((50, -10), -1e-9)

    V = rb_sor(plane, tol=1e-10)
    dV_dy, dV_dx = np.gradient(V, plane.dy, plane.dx)

    Ex = -dV_dx
    Ey = -dV_dy

    '''
    print()
    for tol in [1e-3, 1e-4, 1e-5, 1e-6]:
        V_j = jacobi(plane, tol=tol)
        V_gs = gs(plane, tol=tol)
        V_sor = sor(plane, tol=tol)

        print(f"\ntol = {tol}")
        print("J-best:", np.max(np.abs(V_j - V)))
        print("GS-best:", np.max(np.abs(V_gs - V)))
        print("SOR-best:", np.max(np.abs(V_sor - V)))
        print()
    '''

    fig, (ax_V, ax_E) = plt.subplots(1, 2)

    ax_V.set_title("V")
    draw_grid(ax_V, V, plane)

    ax_E.set_title("E")
    draw_streamplot(ax_E, (Ex, Ey,), plane)

    plt.show()

if __name__ == "__main__":
    main()