import numpy as np
import visualizer as vis

# ======================
# Helper Functions
# ======================

def _find_coefficients(x, y, i, j):
    """
    Compute finite difference coefficients for a grid point.

    Parameters:
    x, y (ndarray): 2D mesh grid coordinates.
    i, j (int): Indices of the grid point.

    Returns:
    Tuple of coefficients (alpha, beta, gamma, delta).
    """
    alpha = 0.25 * ((x[i, j+1] - x[i, j-1])**2 + (y[i, j+1] - y[i, j-1])**2)
    beta = 0.25 * ((x[i+1, j] - x[i-1, j]) * (x[i, j+1] - x[i, j-1]) +
                   (y[i+1, j] - y[i-1, j]) * (y[i, j+1] - y[i, j-1]))
    gamma = 0.25 * ((x[i+1, j] - x[i-1, j])**2 + (y[i+1, j] - y[i-1, j])**2)
    delta = 1/16 * ((x[i+1, j] - x[i-1, j]) * (y[i, j+1] - y[i, j-1]) -
                    (x[i, j+1] - x[i, j-1]) * (y[i+1, j] - y[i-1, j]))

    return alpha, beta, gamma, delta


def _compute_pq(Nx, Ny, zeta_e, ae, ce, zeta_m, eta_m, bm, dm):
    """
    Compute source terms P and Q for elliptic PDE.

    Parameters:
    Nx, Ny (int): Grid dimensions.
    zeta_e, ae, ce (list): Edge source parameters.
    zeta_m, eta_m, bm, dm (list): Point source parameters.

    Returns:
    P, Q (ndarray): Source term arrays.
    """
    P = np.zeros((Nx, Ny))
    Q = np.zeros((Nx, Ny))

    for i in range(Nx):
        for j in range(Ny):

            for l in range(len(zeta_e)):
                dx = i - zeta_e[l]
                P[i, j] += -ae[l] * np.sign(dx) * np.exp(-ce[l] * np.abs(dx))

                de = j - zeta_e[l]
                sign_de = np.sign(de)
                Q[i, j] += ae[l] * sign_de * np.exp(-ce[l] * abs(de))

            for m in range(len(zeta_m)):
                dz = i - zeta_m[m]
                de = j - eta_m[m]
                r = np.sqrt(dz**2 + de**2)
                if r != 0:
                    P[i, j] += -bm[m] * np.sign(de) * np.exp(-dm[m] * r)
                    Q[i, j] += bm[m] * np.sign(dz) * np.exp(-dm[m] * r)

    return P, Q


def _compute_rhs(z, i, j, alpha, beta, gamma, delta, Pij, Qij):
    """
    Compute right-hand side value of SOR iteration.

    Parameters:
    z (ndarray): 2D array (x or y grid).
    i, j (int): Grid point indices.
    alpha, beta, gamma, delta (float): Coefficients.
    Pij, Qij (float): Source terms at (i, j).

    Returns:
    rhs (float): Computed RHS value.
    """
    A = alpha * (z[i-1, j] + z[i+1, j])
    B = -0.5 * beta * (z[i+1, j+1] - z[i-1, j+1] - z[i+1, j-1] + z[i-1, j-1])
    C = gamma * (z[i, j-1] + z[i, j+1])
    D = 0.5 * delta * (Pij * (z[i+1, j] - z[i-1, j]) + Qij * (z[i, j+1] - z[i, j-1]))
    denom = 2 * (alpha + gamma)

    rhs = (A + B + C + D) / denom
    return rhs


def _compute_lhs(z, i, j, alpha, beta, gamma, delta, Pij, Qij):
    """
    Compute residual (LHS) for convergence check.

    Parameters are the same as _compute_rhs.

    Returns:
    lhs (float): Residual value at (i, j).
    """
    A = alpha * (z[i-1, j] - 2 * z[i, j] + z[i+1, j])
    B = -0.5 * beta * (z[i+1, j+1] - z[i-1, j+1] - z[i+1, j-1] + z[i-1, j-1])
    C = gamma * (z[i, j-1] - 2 * z[i, j] + z[i, j+1])
    D = 0.5 * delta * (Pij * (z[i+1, j] - z[i-1, j]) + Qij * (z[i, j+1] - z[i, j-1]))

    lhs = A + B + C + D
    return lhs


def _generate_outer_field(L, n_points, le_dis_deg):
    """
    Generate circular outer boundary with a leading edge displacement.

    Parameters:
    L (float): Radius of the outer field.
    n_points (int): Number of boundary points.
    le_dis_deg (float): Displacement angle in degrees.

    Returns:
    x_outer, y_outer (ndarray): Coordinates of the outer field.
    """
    le_dis_rad = np.deg2rad(le_dis_deg)
    spacing = 2 * np.pi / n_points
    if le_dis_rad > spacing:
        raise ValueError("Displacement angle too large for point spacing")

    theta = np.linspace(0, 2 * np.pi, n_points - 1, endpoint=False)
    theta1 = le_dis_rad
    theta2 = -le_dis_rad % (2 * np.pi)

    x1, y1 = L * np.cos(theta1), L * np.sin(theta1)
    x2, y2 = L * np.cos(theta2), L * np.sin(theta2)

    theta = np.delete(theta, 0)
    x_outer = L * np.cos(theta)
    y_outer = L * np.sin(theta)

    x_outer = np.insert(x_outer, 0, x1)
    y_outer = np.insert(y_outer, 0, y1)
    x_outer = np.append(x_outer, x2)
    y_outer = np.append(y_outer, y2)

    vis.visualize_points(x_outer, y_outer)

    return x_outer, y_outer


# ======================
# Public Functions
# ======================

def initialize_grid(xb, yb, Nx, Ny, L, le_dis_deg, x_dis):
    """
    Initialize computational grid between body and outer field.

    Parameters:
    xb, yb (ndarray): Coordinates of the body.
    Nx, Ny (int): Grid size in x and y directions.
    L (float): Radius of outer circular field.
    le_dis_deg (float): Leading edge displacement angle.
    x_dis (float): Body x-displacement.

    Returns:
    x, y (ndarray): Initialized mesh grid.
    """
    xb = xb - x_dis
    for i in range(len(xb)):
        if xb[i] > L / 2 or xb[i] < -L / 2:
            raise ValueError("Body x-size exceeds outer field size or displacement is too large.")
    for j in range(len(yb)):
        if yb[j] > L / 2 or yb[j] < -L / 2:
            raise ValueError("Body y-size exceeds outer field")

    x = np.zeros((Nx, Ny))
    y = np.zeros((Nx, Ny))

    x_outer, y_outer = _generate_outer_field(L, Nx, le_dis_deg)

    for i in range(Nx):
        x[i, 0] = xb[i]
        y[i, 0] = yb[i]

        x[i, -1] = x_outer[i]
        y[i, -1] = y_outer[i]

        for j in range(1, Ny - 1):
            s = j / (Ny - 1)
            x[i, j] = (1 - s) * x[i, 0] + s * x[i, -1]
            y[i, j] = (1 - s) * y[i, 0] + s * y[i, -1]

    return x, y


def solve_poissons_sor(w, Nx, Ny, x, y, zeta_e, ae, ce, zeta_m, eta_m, bm, dm):
    """
    Solve Poisson’s equation using Successive Over-Relaxation (SOR).

    Parameters:
    w (float): Relaxation factor.
    Nx, Ny (int): Grid dimensions.
    x, y (ndarray): Initial mesh grid.
    zeta_e, ae, ce (list): Edge source parameters.
    zeta_m, eta_m, bm, dm (list): Point source parameters.

    Returns:
    x, y (ndarray): Updated mesh grid.
    n_arr, r_arr (ndarray): Iteration and residual history.
    """
    r = 1
    n = 0

    p, q = _compute_pq(Nx, Ny, zeta_e, ae, ce, zeta_m, eta_m, bm, dm)

    n_arr = []
    r_arr = []

    while r > 1e-6:
        n += 1
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                x_old = x[i, j]
                y_old = y[i, j]
                alpha, beta, gamma, delta = _find_coefficients(x, y, i, j)
                x_new = _compute_rhs(x, i, j, alpha, beta, gamma, delta, p[i, j], q[i, j])
                y_new = _compute_rhs(y, i, j, alpha, beta, gamma, delta, p[i, j], q[i, j])
                x[i, j] = (1 - w) * x_old + w * x_new
                y[i, j] = (1 - w) * y_old + w * y_new

        r = 0
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                alpha, beta, gamma, delta = _find_coefficients(x, y, i, j)
                rx = _compute_lhs(x, i, j, alpha, beta, gamma, delta, p[i, j], q[i, j])
                ry = _compute_lhs(y, i, j, alpha, beta, gamma, delta, p[i, j], q[i, j])
                r += (np.abs(rx) + np.abs(ry)) / 2

        r_avg = r / ((Nx - 1) * (Ny - 1))
        r_arr = np.append(r_arr, r_avg)
        n_arr = np.append(n_arr, n)
        print(f"n: {n} R: {r_avg}")

    return x, y, n_arr, r_arr
