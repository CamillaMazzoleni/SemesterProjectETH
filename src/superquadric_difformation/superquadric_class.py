import numpy as np
from src.utils.math_utils import fexp
from src.superquadric_difformation.superquadrics_plotting import plot_surface, plot_atlas, plot_surface_no_grid, plot_cube_map
from src.superquadric_difformation.difformation_functions import apply_global_bending, apply_circular_bending, apply_parabolic_bending, apply_gaussian_weighted_bending, apply_corner_beding, apply_global_linear_tapering, apply_global_exponential_tapering, apply_global_axial_twisting

class SuperQuadrics:
    def __init__(self, size, shape, resolution=100):
        self.a1, self.a2, self.a3 = size
        self.e1, self.e2 = shape
        self.N = resolution
        self.x, self.y, self.z, self.eta, self.omega = self.sample_equal_distance_on_sq()

    def sq_surface(self, eta, omega):
        x = self.a1 * fexp(np.cos(eta), self.e1) * fexp(np.cos(omega), self.e2)
        y = self.a2 * fexp(np.cos(eta), self.e1) * fexp(np.sin(omega), self.e2)
        z = self.a3 * fexp(np.sin(eta), self.e1)
        return x, y, z

    def sample_equal_distance_on_sq(self):
        eta = np.linspace(-np.pi / 2, np.pi / 2, self.N)
        omega = np.linspace(-np.pi, np.pi, self.N)
        eta, omega = np.meshgrid(eta, omega)
        x, y, z = self.sq_surface(eta, omega)
        return x, y, z, eta, omega

    def apply_tapering(self, ty, tz, method="linear"):
        if method == "linear":
            self.x, self.y, self.z =  apply_global_linear_tapering(self.x, self.y, self.z, ty, tz, self.a1)
            return self.x, self.y, self.z
        elif method == "exponential":
            self.x, self.y, self.z = apply_global_exponential_tapering(self.x, self.y, self.z, ty, tz, self.a1)
            return self.x, self.y, self.z
        else:
            raise ValueError(f"Unknown tapering method: {method}")

    def apply_twisting(self, n):
        self.x, self.y, self.z = apply_global_axial_twisting(self.x, self.y, self.z, n, self.a1)
        return self.x, self.y, self.z

    def apply_bending(self, n, method="global_sinusoidal", center=None, sigma=None, radius=None, factor=None):
        if method == "global_sinusoidal":
            self.x, self.y, self.z = self.apply_global_bending(n)
            return self.x, self.y, self.z
        elif method == "circular":
            self.x, self.y, self.z = self.apply_circular_bending(radius)
            return self.x, self.y, self.z
        elif method == "parabolic":
            self.x, self.y, self.z = self.apply_parabolic_bending(factor)
            return self.x, self.y, self.z
        elif method == "gaussian":
            self.x, self.y, self.z = self.apply_gaussian_weighted_bending(n, center, sigma, factor)
            return self.x, self.y, self.z
        else:
            raise ValueError(f"Unknown bending method: {method}")

    def apply_global_bending(self, n):
        self.x, self.y, self.z = apply_global_bending(self.x, self.y, self.z, self.a1, self.a2, self.a3, n)
        return self.x, self.y, self.z

    def apply_circular_bending(self, radius):
        self.x, self.y, self.z = apply_circular_bending(self.x, self.y, self.z, self.a1, self.a2, self.a3, radius)
        return self.x, self.y, self.z
    

    def apply_parabolic_bending(self, factor):
        self.x, self.y, self.z = apply_parabolic_bending(self.x, self.y, self.z, self.a1, self.a2, self.a3, factor)
        return self.x, self.y, self.z

    def apply_gaussian_weighted_bending(self, n, center, sigma, factor):
        self.x, self.y, self.z = apply_gaussian_weighted_bending(self.x, self.y, self.z, center, sigma, factor)
        return self.x, self.y, self.z
    
    def apply_corner_bending(self, bend_factor):
        vertices = np.stack((self.x.flatten(), self.y.flatten(), self.z.flatten()), axis=-1)
        bent_vertices = np.array([apply_corner_beding(x, y, z, bend_factor) for x, y, z in vertices])
        self.x = bent_vertices[:, 0].reshape(self.x.shape)
        self.y = bent_vertices[:, 1].reshape(self.y.shape)
        self.z = bent_vertices[:, 2].reshape(self.z.shape)
        return self.x, self.y, self.z


    def plot(self, ax, point=None, title="Superquadric"):
        plot_surface_no_grid(ax, self.x, self.y, self.z, point, title)

    def plot_atlas(self, ax, magnitudes, title, withgrid= False, withlabels= False):
        return plot_atlas(ax, magnitudes, title, withgrid, withlabels)

    def plot_cube_map(self, title="Cube Map"):
        return plot_cube_map(self, self.x, self.y, self.z, title)
    

    def select_random_point(self):
        idx_eta = np.random.randint(0, self.eta.shape[0])
        idx_omega = np.random.randint(0, self.omega.shape[1])
        point = np.array([self.x[idx_eta, idx_omega], self.y[idx_eta, idx_omega], self.z[idx_eta, idx_omega]])
        return point
