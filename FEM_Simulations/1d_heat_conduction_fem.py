# 1D Heat Conduction in a Fiber using FEM
# ----------------------------------------
# Problem Description:
# This program solves the 1D heat conduction equation along a fiber using the Finite Element Method (FEM).
# The governing PDE is:
#     -d/dx [k(x) * dT/dx] = q(x), for x in [0, L],
# where:
#     T(x) = temperature distribution (unknown we solve for),
#     k(x) = thermal conductivity of the fiber (W/m·K),
#     q(x) = heat source term (W/m³),
#     L = length of the fiber (m).
#
# Boundary Conditions:
# 1. Dirichlet Condition: Fixed temperature at the left boundary:
#        T(0) = 100°C.
# 2. Neumann Condition: Fixed heat flux at the right boundary:
#        -k * dT/dx |_{x=L} = 50 W/m².
#
# Material and Setup:
# - Fiber length: L = 1.0 m.
# - Thermal conductivity: k = 200 W/m·K (constant).
# - Uniform heat generation: q = 500 W/m³.
#
# Steps:
# 1. Divide the fiber into n = 10 elements.
# 2. Assemble the stiffness matrix (K) and force vector (F) using FEM principles.
# 3. Apply boundary conditions:
#        a. Dirichlet at x=0 (fixed temperature).
#        b. Neumann at x=L (heat flux).
# 4. Solve the linear system (K * T = F) to find the nodal temperatures.
# 5. Plot the temperature distribution along the fiber.
#
# Output:
# - Temperature values at each node along the fiber.
# - Plot of temperature distribution.


import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Parameters
L = 1.0 #Length of fiber (m)
n = 10 # Number of finite Elements we will use to discretize the domain
k = 200 # Thermal Conductivity (W/m*k)
q = 500 # Heat Source (W/m^3) or the external source term 
h = L/n # Element size (m), this is the spacing between nodes in the domain

# Node positions
x = np.linspace(0, L, n+1) # the unknowns (u(x)) are solved on these nodes 

# Initialize global stiffness matrix and force vector
K = np.zeros((n+1, n+1))  # Stiffness matrix
F = np.zeros(n+1)         # Force vector

# Assemble the stiffness matrix and force vector
for i in range(n):
    # Stiffness matrix contributions
    K[i, i] += k / h
    K[i, i+1] -= k / h
    K[i+1, i] -= k / h
    K[i+1, i+1] += k / h
    
    # Force vector contributions (heat source)
    F[i] += q * h / 2
    F[i+1] += q * h / 2

# Apply boundary conditions
# Dirichlet condition: T(0) = 100°C
K[0, :] = 0
K[:, 0] = 0
K[0, 0] = 1
F[0] = 100  # Fixed temperature at x=0

# Neumann condition: Heat flux at x=L
F[-1] += 50  # Add heat flux contribution to the last node


# Print the stiffness matrix and force vector AFTER applying boundary conditions
print("\nGlobal stiffness matrix (K) after applying boundary conditions:")
print(K)
print("\nGlobal force vector (F) after applying boundary conditions:")
print(F)


# Solve the system of equations
T = scipy.linalg.solve(K, F)

# Print the solution
print("\nTemperature distribution (T):")
print(T)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(x, T, marker='o', label="Temperature Distribution")
plt.xlabel('Position along the fiber (m)')
plt.ylabel('Temperature (°C)')
plt.title('1D Heat Conduction in a Fiber')
plt.legend()
plt.grid()
plt.show()