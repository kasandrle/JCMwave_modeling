import scipy.constants as const
import numpy as np
#import numba
from scipy.interpolate import interp1d

# Precompute hc in nm·eV
hc = const.h * const.c / const.e * 1e9  # Planck × speed of light / charge × 1e9

def eVnm_converter(value):
    """Convert photon energy in eV to wavelength in nm."""
    return hc / value

def load_nk_from_file(filepath,energy_pol_uni):
    energy_uni = []
    for label in energy_pol_uni:
        energy_str, pol = label.split('_')
        energy = float(energy_str)
        energy_uni.append(energy)
    df = pd.read_csv(filepath)
    e_arr = np.array(df['Energy'])
    n_arr = np.array(df['delta'])
    k_arr = np.array(df['beta'])
    n_interp = interp1d(e_arr, n_arr, fill_value="extrapolate")
    k_interp = interp1d(e_arr, k_arr, fill_value="extrapolate")

    n_array_extended = n_interp(energy_uni)
    k_array_extended = k_interp(energy_uni)
    return n_array_extended, k_array_extended

#@numba.jit(nopython=True, cache=True, fastmath=True, nogil=True)
def corner_round(x1, x2, x3, r, n=50):
    """
    Calculates the x and y points of a rounded corner
    :param x1: 2d array with x,y of point 1
    :param x2: 2d array with x,y of point 2 (this is the point in the middle)
    :param x3: 2d array with x,y of point 3
    :param r: radius of the corner
    :param n: number of points of the corner
    :return: array containing the x,y of the rounded corner
    """
    #direction vector from x2 to x1 and x2 to x3
    a = np.empty(2)
    b = np.empty(2)

    a[0] = x2[0] - x1[0]
    a[1] = x2[1] - x1[1]
    b[0] = x2[0] - x3[0]
    b[1] = x2[1] - x3[1]

    # norm of a and b
    norme_a = np.linalg.norm(a)
    norme_b = np.linalg.norm(b)
    # normded vector a and b
    norm_a = a / norme_a
    norm_b = b / norme_b
    # angles between a and b
    ang = np.arccos(np.dot(a, b) / norme_a / norme_b)
    # angle of the a vector to the x-axis, needed as an offset
    ang_0 = np.arccos(a[0] / norme_a)
    # angular apertur of the corner
    beta2 = np.pi - ang
    # distance between x2 and the center of the circle of the corner and x2
    c = -r / np.sin(ang / 2)
    # vector pointing along the line between the center of the circle of the corner and x2
    direct = (norm_a + norm_b) / np.linalg.norm(norm_a + norm_b)
    # calculating the x and y of the center of the circle
    c_point = x2 + c * direct
    # look in wich part of the circle the corner need to be
    if direct[0] > 0 and direct[1] > 0:
        rot = -1
    elif direct[0] < 0 and direct[1] < 0:
        rot = 0
    elif direct[0] < 0 and direct[1] > 0:
        rot = -1
    else:
        rot = 2

    result = np.empty((n, 2))
    # angle offset
    corr_ang = rot / 2 * np.pi + ang_0
    # calculate the points of the corner
    for i in range(n):
        result[i][0] = np.cos(beta2 / n * i + corr_ang) * r + c_point[0]
        result[i][1] = np.sin(beta2 / n * i + corr_ang) * r + c_point[1]
    # look if the points are in the right order
    if a[0] * b[1] - a[1] * b[0] > 0:
        result = result[::-1]

    return result