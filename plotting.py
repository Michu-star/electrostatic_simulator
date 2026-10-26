def draw_grid(ax, grid, plane):
    image = ax.imshow(grid,
                      origin="lower",
                      extent=[0, plane.width, 0, plane.height])
    ax.figure.colorbar(image, ax=ax, label="Potential V")
    ax.set(xlabel="x", ylabel="y")


def draw_streamplot(ax, streamplot, plane):
    ax.streamplot(plane.x, plane.y, *streamplot, color="black")
    ax.set_aspect("equal")
