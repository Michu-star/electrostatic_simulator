import numpy as np
import matplotlib.pyplot as plt

grid = np.zeros((100,100))

# Set boundaries
grid[:, 0] = 1.0
grid[:, -1] = -1.0

for i in range(500):
    for j in range(1, grid.shape[0] - 1):
        for k in range(1, grid.shape[1] - 1):
            grid[j,k] = 0.25 * (
                grid[j + 1, k] + grid[j - 1, k]
                + grid[j, k + 1] + grid[j, k - 1]
            )


plt.imshow(grid, origin="lower")
plt.colorbar(label="Potential V")
plt.show()