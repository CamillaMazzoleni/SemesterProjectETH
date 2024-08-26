import numpy as np

def apply_global_bending(x, y, z, a1, a2, a3, n):
    X = x
    Y = y
    Z = a3 * np.sin(n * np.pi * x / a1) + z
    return X, Y, Z

def apply_circular_bending(x, y, z, a1, a2, a3, radius):
    theta = x / radius
    X = radius * np.sin(theta)
    Y = y
    Z = radius * (1 - np.cos(theta)) + z
    return X, Y, Z

def apply_parabolic_bending(x, y, z, a1, a2, a3, factor):
    X = x
    Y = y
    Z = z + factor * (x**2 / a1)
    return X, Y, Z


def apply_gaussian_weighted_bending(x, y, z, center, sigma, factor):
    # Compute the bending parameters from the bending vector
    k = factor[0]
    a = factor[1]
    b = factor[2]
    r0, z0 = center

    # Compute cylindrical coordinates
    R = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Gaussian weighting function for localized bending
    w = np.exp(-((R - r0)**2 / (2 * sigma[0]**2) + (z - z0)**2 / (2 * sigma[1]**2)))

    # Apply the bending transformation
    z_bend = k * (R - a) * np.cos(b * theta)

    x_d = x
    y_d = y
    z_d = z + w * z_bend

    return x_d, y_d, z_d

def apply_corner_beding(x, y, z, bend_factor):
    x_bend = x * (1 + bend_factor[0] * (x**2 + y**2 + z**2))
    y_bend = y * (1 + bend_factor[1] * (x**2 + y**2 + z**2))
    z_bend = z * (1 + bend_factor[2] * (x**2 + y**2 + z**2))
    return x_bend, y_bend, z_bend

def apply_global_linear_tapering(x, y, z, ty, tz, a1):
    Y = (((ty / a1) * x) + 1) * y
    Z = (((tz / a1) * x) + 1) * z
    return x, Y, Z

def apply_global_exponential_tapering(x, y, z, ty, tz, a1):
    Y = np.exp((ty / a1) * x) * y
    Z = np.exp((tz / a1) * x) * z
    return x, Y, Z

def apply_global_axial_twisting(x, y, z, n, a1):
    theta = n * (np.pi + np.pi * x / a1)
    Y = np.cos(theta) * y - np.sin(theta) * z
    Z = np.sin(theta) * y + np.cos(theta) * z
    return x, Y, Z
