import visualizer as vis
import mesh_generator as mg
import coefficient_generator as cg
import airfoil_boundary_generator as abg


if __name__ == "__main__":
    # === Step 1: Generate Airfoil Boundary ===
    Nx = 11  # Number of points along x-direction (airfoil chord)
    Ny = 5   # Number of points in the y-direction (normal to airfoil)
    naca_number = "2412"

    xb, yb = abg.generate_naca_airfoil(naca_number, Nx)
    vis.visualize_points(xb, yb)  # Plot airfoil boundary points

    # === Step 2: Initialize Grid ===
    L = 2            # Length of the domain downstream
    le_dis_deg = 1   # Leading edge displacement in degrees (rotation)
    x_dis = 0        # X-displacement of the airfoil

    # Generate initial grid based on airfoil surface and domain settings
    x, y = mg.initialize_grid(xb, yb, Nx, Ny, L, le_dis_deg, x_dis)
    vis.plot_grid(x, y, "Mesh Grid Before SOR")

    # === Step 3: Source Terms and Relaxation Setup ===
    # Define source control parameters (elliptic PDE coefficients)
    zeta_e = cg.get_1D_array_zero_to_one(Nx)  # Distribution of zeta along the airfoil
    ae = cg.constant_1D_array(Nx, 5)          # a(zeta)
    ce = cg.constant_1D_array(Nx, 10)         # c(zeta)

    # Define sink/source control points in the domain
    zeta_m = [0]      # zeta location of source
    eta_m = [0.5]     # eta location of source
    bm = [12.0]       # source strength
    dm = [4.0]        # decay factor

    w = 1.4  # Relaxation factor for SOR (0 < w < 2 for convergence)

    # Solve Poisson's equation using Successive Over-Relaxation (SOR)
    x, y, n, R = mg.solve_poissons_sor(
        w, Nx, Ny, x, y,
        zeta_e, ae, ce,
        zeta_m, eta_m, bm, dm
    )

    # === Step 4: Visualization ===
    vis.plot_grid(x, y, "Mesh Grid After SOR")  # Plot updated mesh
    vis.visualize_points_ylog(n, R, "Residual vs Iteration", "Iterations", "Residual")
