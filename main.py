import numpy as np
import matplotlib.pyplot as plt

from physics import ElectrostaticSystem, jacobi, rb_sor
from plotting import draw_grid, draw_streamplot


def main():
    plane = ElectrostaticSystem((201, 201), 5, 10)

    # Positive electrode
    plane.add_square_electrode(0.5, 1, 4.5, 1 + plane.dy, 200)

    # Negative electrode
    plane.add_square_electrode(0.5, 9, 4.5, 9 + plane.dy, -200)

    # Point charges
    plane.add_point_charge(4, 6, 1e-9)

    # Make edges zero
    plane.set_potential((0, slice(None)), 0)
    plane.set_potential((-1, slice(None)), 0)
    plane.set_potential((slice(None), 0), 0)
    plane.set_potential((slice(None), -1), 0)

    V = rb_sor(plane, tol=1e-6)
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