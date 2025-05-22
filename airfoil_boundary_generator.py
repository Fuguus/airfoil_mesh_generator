import numpy as np


# Helper Function
def _camber_line(x, cmax, pmax):
    """
    Computes the camber line (mean line) and its slope for a NACA 4-digit airfoil.

    Parameters:
    x (ndarray): x-coordinates along the chord (0 to 1).
    cmax (float): Maximum camber.
    pmax (float): Position of maximum camber.

    Returns:
    tuple: camber line y-values (yc), and cosine and sine of slope angle (theta).
    """
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    for i, xi in enumerate(x):
        if xi < pmax:
            yc[i] = (cmax / pmax**2) * (2 * pmax * xi - xi**2)
            dyc_dx[i] = (2 * cmax / pmax**2) * (pmax - xi)
        else:
            yc[i] = (cmax / (1 - pmax)**2) * ((1 - 2 * pmax) + 2 * pmax * xi - xi**2)
            dyc_dx[i] = (2 * cmax / (1 - pmax)**2) * (pmax - xi)

    theta = np.arctan(dyc_dx)
    return yc, np.cos(theta), np.sin(theta)


def _thickness_dist(x, tmax):
    """
    Computes the airfoil thickness distribution for a NACA 4-digit airfoil.

    Parameters:
    x (ndarray): x-coordinates along the chord.
    tmax (float): Maximum thickness.

    Returns:
    ndarray: Thickness distribution along the chord.
    """
    x0 = 1.008930 * x  # Empirical correction factor for better profile match
    yt = (0.29690 * np.sqrt(x0)
          - 0.12600 * x0
          - 0.35160 * x0**2
          + 0.28430 * x0**3
          - 0.10150 * x0**4)
    return (tmax / 0.20) * yt


# Main Function
def generate_naca_airfoil(naca, nnode):
    """
    Generates the boundary coordinates of a NACA 4-digit airfoil.

    Parameters:
    naca (str): 4-digit NACA airfoil number as string (e.g., "2412").
    nnode (int): Total number of points to generate (must be odd and >= 5).

    Returns:
    tuple: x and y coordinates of the airfoil boundary.
    """
    if nnode < 5:
        raise ValueError("More number of points are required to generate an airfoil.")
    if nnode % 2 == 0:
        raise ValueError("Only odd number of points can be generated for the airfoil.")
    if len(naca) != 4 or not naca.isdigit():
        raise ValueError("NACA number must be a 4-digit string.")

    # Parse NACA parameters
    cmax = int(naca[0]) / 100.0  # maximum camber
    pmax = int(naca[1]) / 10.0   # position of maximum camber
    tmax = int(naca[2:]) / 100.0 # maximum thickness

    npanel = nnode - 1
    theta = np.linspace(0, np.pi, npanel // 2 + 1)
    x_cosine = 0.5 * (1 - np.cos(theta))  # cosine spacing

    x = np.zeros(nnode)
    y = np.zeros(nnode)

    yc, cth, sth = _camber_line(x_cosine, cmax, pmax)
    yt = _thickness_dist(x_cosine, tmax)

    # Lower and upper surfaces
    xl = x_cosine + yt * sth
    yl = yc - yt * cth

    xu = x_cosine - yt * sth
    yu = yc + yt * cth

    # Combine upper and lower surfaces
    x = np.concatenate([xu[::-1], xl[1:]])
    y = np.concatenate([yu[::-1], yl[1:]])

    # Normalize x so that it starts at 0 and ends at 1
    x -= x.min()
    x /= x.max()

    return x, y
