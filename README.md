﻿# Fiber Simulation Project
# Fiber Simulation Project

## Overview

The **Fiber Simulation Project** is a collaborative effort to model, simulate, and analyze the physical behavior of fibers under various conditions. The project aims to combine **Finite Element Methods (FEM)**, **Computational Fluid Dynamics (CFD)**, and **Multiphysics simulations** into a cohesive framework, integrating insights from machine learning to tackle advanced simulation challenges. 

This project is being developed collaboratively by:
- **Griffin Thompson**: Undergraduate Data Science student at Yale University.
- **Davin Hickman-Chow**: Biomedical Engineering undergraduate at Washington University in St. Louis.

We leverage state-of-the-art simulation techniques to understand and predict fiber behavior, with applications ranging from **biomedical devices** to **aerospace engineering**.

---

## Goals

1. **Fiber Physics Modeling**:
   - Analyze and simulate the structural and mechanical properties of fibers.
   - Explore fiber deformation, elasticity, and material interactions.

2. **Finite Element Method (FEM)**:
   - Implement FEM solvers for static and dynamic problems.
   - Develop eigenvalue analysis techniques to study fiber stability and vibration.

3. **Computational Fluid Dynamics (CFD)**:
   - Model fiber interactions with fluid flows, including stationary fibers and turbulent environments.
   - Simulate airflow and boundary conditions around fibers using finite volume methods.

4. **Multiphysics Simulations**:
   - Combine FEM and CFD to model coupled physics scenarios.
   - Extend simulations to real-world biomedical applications, such as fiber-reinforced materials in medical devices.

5. **Machine Learning Integration**:
   - Integrate machine learning models to optimize simulation workflows and predict fiber behavior.
   - Develop ML-assisted parameter tuning and data-driven modeling frameworks.

6. **Culmination**:
   - Demonstrate the project’s capabilities through a comprehensive **fiber simulation model**, integrating lessons from all domains.

---

## Project Structure
📂 FiberSimulationProject
├── 📂 Fiber_Physics
│   ├── README.md                 # Overview of fiber properties (Chapters from Fiber Technology)
│   ├── 📂 PhysicalModels          # Python scripts for fiber elasticity, strength, and material interactions
│
├── 📂 FEM_Simulations
│   ├── README.md                 # Explanation of FEM projects
│   ├── 📂 Chapter1_Basics         # Simple FEM stiffness matrix solvers
│   ├── 📂 Dynamic_Problems        # Newmark, Trapezoidal algorithms
│   └── 📂 Eigenvalue_Analysis     # Scripts for eigenvalue problems and stability analysis
│
├── 📂 CFD_Simulations
│   ├── README.md                 # Explanation of CFD workflows
│   ├── 📂 Stationary_Fibers       # Stationary fiber in fluid flow (Simple Finite Volume models)
│   ├── 📂 Turbulence_Models       # k-ε and k-ω turbulence models
│   └── 📂 FiberFluidInteraction   # Simulations for fibers interacting with airflow
│
├── 📂 Multiphysics_Simulations
│   ├── README.md                 # Overview of Multiphysics simulations
│   ├── 📂 COMSOL_Workflows        # COMSOL simulation files and workflows
│   ├── 📂 Coupled_Simulations     # FEM + CFD integration simulations
│   └── 📂 Biomedical_Applications # Advanced applications for fiber-reinforced biomedical devices
│
├── 📂 Docs
│   ├── Progress_Notes.md         # Notes on project progress
│   ├── Chapter_Summaries.md      # Summaries of book chapters for reference
│   ├── Resources.md              # Reference materials, links, and guides
│   └── Coding_Guide.md           # Coding standards, tips, and tools used in the project
│
├── 📂 Tests
│   ├── 📂 FEM_Tests              # Scripts for validating FEM implementations
│   ├── 📂 CFD_Tests              # Scripts for validating CFD models
│   └── 📂 Multiphysics_Tests     # Scripts for testing coupled simulations
│
└── README.md                     # Main overview of the project


---

## Book References and Goals

The project is informed by the following key texts. Each book contributes to a distinct area of the project:

1. **Fiber Technology**:
   - **Goals**: Understand fiber properties, material compositions, and processing methods.
   - **Focus**: Chapters on physical properties, manufacturing, and deformation analysis.

2. **The Finite Element Method (Thomas J.R. Hughes)**:
   - **Goals**: Build FEM solvers for static, dynamic, and eigenvalue problems.
   - **Focus**: Chapters on stiffness matrix assembly, dynamic solvers (Newmark, Trapezoidal), and eigenvalue analysis.

3. **Notes on Computational Fluid Dynamics**:
   - **Goals**: Develop CFD solvers for fiber-fluid interactions.
   - **Focus**: Boundary conditions, turbulence models (k-ε, k-ω), and sample problems for stationary fibers.

4. **Multiphysics Modeling Using COMSOL**:
   - **Goals**: Combine FEM and CFD in coupled simulations for biomedical and industrial applications.
   - **Focus**: Workflow optimization, COMSOL scripting, and heat transfer in fiber-reinforced composites.

---

## Collaboration Workflow

### 1. **Setting Up the Repository**
- Clone the repository:
  ```bash
  git clone https://github.com/GriffThomp1234/FiberSimulationProject.git
  cd FiberSimulationProject

