import scipy.constants as const
import numpy as np
#import numba
import pandas as pd
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

def arc_from_three_points(p1, p2, pc, n=50):
    """
    Generates an arc from p1 to p2 around center pc.
    :param p1: start point (x, y)
    :param p2: end point (x, y)
    :param pc: center of the circle (x, y)
    :param n: number of points
    :return: (n, 2) array of arc coordinates
    """
    # Vectors from center to points
    v1 = np.array(p1) - np.array(pc)
    v2 = np.array(p2) - np.array(pc)

    # Angles from center to points
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])

    # Determine sweep direction using cross product
    cross = np.cross(v1, v2)
    if cross < 0:
        # Clockwise
        if angle1 < angle2:
            angle1 += 2 * np.pi
        angles = np.linspace(angle1, angle2, n)
    else:
        # Counterclockwise
        if angle2 < angle1:
            angle2 += 2 * np.pi
        angles = np.linspace(angle1, angle2, n)

    radius = np.linalg.norm(v1)
    arc = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius + np.array(pc)
    return arc

def circle_line_intersection(center, radius, p_start, p_end):
    p1 = np.array(p_start) - center
    p2 = np.array(p_end) - center
    d = p2 - p1
    dr2 = np.dot(d, d)
    D = p1[0]*p2[1] - p2[0]*p1[1]
    discriminant = radius**2 * dr2 - D**2

    if discriminant < 0:
        return []

    sqrt_disc = np.sqrt(discriminant)
    sign_dy = np.sign(d[1]) if d[1] != 0 else 1

    x1 = (D * d[1] + sign_dy * d[0] * sqrt_disc) / dr2
    y1 = (-D * d[0] + abs(d[1]) * sqrt_disc) / dr2
    x2 = (D * d[1] - sign_dy * d[0] * sqrt_disc) / dr2
    y2 = (-D * d[0] - abs(d[1]) * sqrt_disc) / dr2

    pt1 = np.array([x1, y1]) + center
    pt2 = np.array([x2, y2]) + center

    def on_segment(p):
        return np.all(np.minimum(p_start, p_end) <= p) and np.all(p <= np.maximum(p_start, p_end))

    return [pt for pt in [pt1, pt2] if on_segment(pt)]

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

    min_leg_length = min(np.linalg.norm(np.array(x2) - np.array(x1)),
                     np.linalg.norm(np.array(x2) - np.array(x3)))
    #print(min_leg_length)
    if r > min_leg_length:
        print('Radius may be to large, shape may look strange')
    #    r = min_leg_length / 2  # prevent overreach


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
    #print(norme_a,norme_b)
    # normded vector a and b
    norm_a = a / norme_a
    norm_b = b / norme_b
    # angles between a and b
    ang = np.arccos(np.dot(a, b) / norme_a / norme_b)
    #print(np.rad2deg(ang))
    # angle of the a vector to the x-axis, needed as an offset
    ang_0 = np.arccos(a[0] / norme_a)
    #print(np.rad2deg(ang_0))
    # angular apertur of the corner
    beta2 = np.pi - ang
    # distance between x2 and the center of the circle of the corner and x2
    c = -r / np.sin(ang / 2)
    # vector pointing along the line between the center of the circle of the corner and x2
    direct = (norm_a + norm_b) / np.linalg.norm(norm_a + norm_b)
    # calculating the x and y of the center of the circle
    c_point = x2 + c * direct

    t1_all = circle_line_intersection(c_point, r, x1, x2)
    t2_all = circle_line_intersection(c_point, r, x2, x3)

    if not t1_all or not t2_all:
        if ang < np.pi/2:
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
        else:
            print('Radius may be to small, returned x2')
            return np.array([x2])

    #print(t1_all[0],t2_all[0])
    result = arc_from_three_points(t1_all[0],t2_all[0],c_point)


    return result

