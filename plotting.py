def draw_grid(ax, grid):
    image = ax.imshow(grid, origin="lower")
    ax.figure.colorbar(image, ax=ax, label="Potential V")