# Airfoil Mesh Generator

A modular Python tool to generate and smooth computational grids around airfoils using source-controlled elliptic mesh generation and Successive Over-Relaxation (SOR) techniques.

## Overview

This project takes a specified NACA airfoil profile and:

1. **Generates boundary points** for the airfoil.
2. **Initializes a mesh grid** in a user-defined flow domain.
3. **Applies source control** and smoothing via Poisson’s equation using the SOR method.
4. **Visualizes** the airfoil, mesh (before and after smoothing), and the residual convergence.

---

## Project Structure

├── main.py # Main script to run the mesh generation process

├── visualizer.py # Contains functions to visualize grids and plots

├── mesh_generator.py # Grid initialization and Poisson solver with SOR

├── coefficient_generator.py # Generates source terms and control functions

├── airfoil_boundary_generator.py# Creates the NACA airfoil shape

---

## ⚙️ Usage

### 🔧 Parameters in `main.py`
- `Nx`: Number of points along the airfoil chord.
- `Ny`: Number of mesh points in the normal direction.
- `naca_number`: 4-digit NACA airfoil to generate.
- `L`: Length of the downstream domain.
- `le_dis_deg`: Rotation angle (deg) of the leading edge.
- `x_dis`: X-offset of the airfoil in the domain.
- `w`: Relaxation factor for the SOR method.

### Run the code

```bash
python main.py
