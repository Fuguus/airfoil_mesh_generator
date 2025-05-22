import numpy as np


def get_1D_array_zero_to_one(n_points):
    """
    Create a 1D NumPy array with values from 0 to n_points - 1.

    Parameters:
    n_points (int): Number of points in the array.

    Returns:
    ndarray: Array of values [0, 1, ..., n_points - 1].
    """
    return np.arange(n_points)


def constant_1D_array(n_points, strength):
    """
    Create a 1D NumPy array of given size filled with a constant value.

    Parameters:
    n_points (int): Number of points in the array.
    strength (float): Value to fill the array with.

    Returns:
    ndarray: Constant-filled array.
    """
    return np.full(n_points, strength)