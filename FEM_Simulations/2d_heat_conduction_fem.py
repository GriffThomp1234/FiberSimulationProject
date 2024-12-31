import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
Lx = 0.1  # Length of the fiber in x-direction (m)
Ly = 0.02  # Length of the fiber in y-direction (m)
nx, ny = 10, 10  # Number of elements in x and y directions
k = 100  # Thermal conductivity (W/m·K)
q = 500  # Heat source (W/m²)
dx = Lx / nx  # Element size in x-direction (m)
dy = Ly / ny  # Element size in y-direction (m)

# Generate mesh
x_nodes = np.linspace(0, Lx, nx+1)  # Node positions in x
y_nodes = np.linspace(0, Ly, ny+1)  # Node positions in y
num_nodes = (nx + 1) * (ny + 1)

# Create node numbering
def node_index(i, j):
    return i * (ny + 1) + j

# Initialize global stiffness matrix and force vector
K = np.zeros((num_nodes, num_nodes))
F = np.zeros(num_nodes)

# Assembly of the stiffness matrix and force vector
for i in range(nx):
    for j in range(ny):
        # Node indices for the current element
        n1 = node_index(i, j)
        n2 = node_index(i+1, j)
        n3 = node_index(i+1, j+1)
        n4 = node_index(i, j+1)

        # Local stiffness matrix for bilinear elements
        Ke = (k / (dx * dy)) * np.array([
            [ 2, -2, -1,  1],
            [-2,  2,  1, -1],
            [-1,  1,  2, -2],
            [ 1, -1, -2,  2]
        ])

        # Local force vector
        Fe = (q * dx * dy / 4) * np.array([1, 1, 1, 1])

        # Assemble into global matrices
        indices = [n1, n2, n3, n4]
        for a in range(4):
            for b in range(4):
                K[indices[a], indices[b]] += Ke[a, b]
            F[indices[a]] += Fe[a]

# Apply boundary conditions
# Dirichlet condition: T = 100 at left boundary (x=0)
for j in range(ny + 1):
    n = node_index(0, j)
    K[n, :] = 0
    K[:, n] = 0
    K[n, n] = 1
    F[n] = 100  # Fixed temperature

# Solve for temperature
T = np.linalg.solve(K, F)

# Reshape T to 2D for plotting
T_2D = T.reshape((nx + 1, ny + 1))

# Plot the temperature distribution
plt.figure(figsize=(8, 6))
plt.contourf(x_nodes, y_nodes, T_2D.T, 20, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('2D Heat Conduction in a Fiber')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()
