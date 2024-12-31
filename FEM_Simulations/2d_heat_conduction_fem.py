# 2D Heat Conduction in a Rectangular Fiber using FEM
# ---------------------------------------------------
# Problem Description:
# This program solves the 2D heat conduction equation in a rectangular fiber using the Finite Element Method (FEM).
# The governing PDE is:
#     -∇⋅[k(x, y) ∇T(x, y)] = q(x, y), for (x, y) ∈ Ω,
# where:
#     T(x, y) = temperature distribution (unknown we solve for),
#     k(x, y) = thermal conductivity of the fiber (W/m·K),
#     q(x, y) = heat source term (W/m²),
#     Ω = 2D domain of the fiber (rectangular shape, size Lx × Ly).
#
# Boundary Conditions:
# 1. Dirichlet Condition: Fixed temperature at the left boundary (x = 0):
#        T(0, y) = 100°C, for y ∈ [0, Ly].
# 2. Neumann Condition: Fixed heat flux at the right boundary (x = Lx):
#        -k ∂T/∂x |_{x=Lx} = q_flux, for y ∈ [0, Ly].
#        (e.g., q_flux = 50 W/m²).
#
# Material and Setup:
# - Fiber dimensions: Lx = 0.1 m (length in x), Ly = 0.02 m (length in y).
# - Thermal conductivity: k = 100 W/m·K (constant).
# - Uniform heat generation: q = 500 W/m².
#
# Steps:
# 1. Divide the 2D domain into nx × ny = 10 × 10 elements.
# 2. Generate the finite element mesh with nodal positions and numbering.
# 3. Assemble the global stiffness matrix (K) and force vector (F) using FEM principles:
#        a. Compute local stiffness matrices for bilinear elements.
#        b. Add contributions to the global matrix/vector.
# 4. Apply boundary conditions:
#        a. Dirichlet at x=0 (fixed temperature on left boundary).
#        b. Neumann at x=Lx (heat flux on right boundary).
# 5. Solve the linear system (K * T = F) to find the nodal temperatures.
# 6. Visualize the temperature distribution as a contour plot.
#
# Output:
# - Temperature values at each node in the 2D grid.
# - Contour plot of temperature distribution over the 2D domain.
#
# Example Application:
# - Simulate heat transfer through a fiber-like material to observe the thermal gradient due to a heat source and specified boundary conditions.

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
