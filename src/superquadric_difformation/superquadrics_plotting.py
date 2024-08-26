import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt


def plot_circular_plane(ax, radius, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    circle_z = np.zeros_like(theta)
    ax.plot(circle_x, circle_y, circle_z, 'b-', alpha=0.5)

def plot_rectangular_plane(ax, size, num_points=100):
    a1, a2, a3 = size
    x = np.linspace(-a1, a1, num_points)
    y = np.linspace(-a2, a2, num_points)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)
    ax.plot_surface(x, y, z, color='b', alpha=0.1)

def plot_angle_arc_y_plane(ax, radius, beta, num_points=100):
    theta_beta = np.linspace(0, beta, num_points)
    x_beta = radius * np.cos(theta_beta)
    y_beta = radius * np.sin(theta_beta)
    z_beta = np.zeros_like(theta_beta)
    ax.plot(x_beta, y_beta, z_beta, 'b-', linewidth=5)

def plot_angle_arc_x_plane(ax, radius, alpha, num_points=100):
    theta_beta = np.linspace(0, alpha, num_points)
    x_beta = np.zeros_like(theta_beta)
    y_beta = radius * np.cos(theta_beta)
    z_beta = radius * np.sin(theta_beta)
    ax.plot(x_beta, y_beta, z_beta, 'r-', linewidth=5)

def plot_angle_semicircle(ax, point, alpha, beta, num_points=100):
    # Semicircle for alpha in the xz-plane
    theta_alpha = np.linspace(0, alpha, num_points)
    x_alpha = point[0] * np.cos(theta_alpha)
    z_alpha = point[0] * np.sin(theta_alpha)
    y_alpha = np.zeros(num_points)
    ax.plot(x_alpha, y_alpha, z_alpha, 'r--')

    # Semicircle for beta in the yz-plane
    theta_beta = np.linspace(0, beta, num_points)
    y_beta = point[1] * np.cos(theta_beta)
    z_beta = point[1] * np.sin(theta_beta)
    x_beta = np.zeros(num_points)
    ax.plot(x_beta, y_beta, z_beta, 'b--')

def plot_surface(ax, x, y, z, point=None, title="Superquadric"):
    ax.plot_surface(x, y, z, color='g', alpha=0.3)
    if point is not None:
        ax.scatter(point[0], point[1], point[2], color='r', s=10)
        ax.quiver(0, 0, 0, point[0], point[1], point[2], color='r', length=np.linalg.norm(point), normalize=True)
    ax.set_xlim([-0.65, 0.65])
    ax.set_ylim([-0.65, 0.65])
    ax.set_zlim([-0.65, 0.65])
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    ax.set_title(title)


def plot_atlas(ax, magnitudes, title = None, withgrid= False, withlabels= False):
    # Normalize magnitudes to the range [0, 1]
    norm = Normalize(vmin=0, vmax=1)
    c = ax.imshow(magnitudes, extent=[-180, 180, -90, 90], origin='lower', cmap='viridis', norm=norm)
    
    if not withgrid:
        ax.axis('off')

    if withlabels:  
        ax.set_xlabel("β (degrees)")
        ax.set_ylabel("α (degrees)")

    if title is not None:
        ax.set_title(title)
    return c


def plot_cube_map(superquadric, x, y, z, title="Cube Map"):
    cube_map_faces = {'right': [], 'left': [], 'top': [], 'bottom': [], 'front': [], 'back': []}
    points = zip(x.flatten(), y.flatten(), z.flatten())
    for x, y, z in points:
        if abs(x) >= abs(y) and abs(x) >= abs(z):
            face = 'right' if x > 0 else 'left'
            u = y / abs(x)
            v = z / abs(x)
        elif abs(y) >= abs(x) and abs(y) >= abs(z):
            face = 'top' if y > 0 else 'bottom'
            u = x / abs(y)
            v = z / abs(y)
        else:
            face = 'front' if z > 0 else 'back'
            u = x / abs(z)
            v = y / abs(z)
        
        u = (u + 1) / 2
        v = (v + 1) / 2
        cube_map_faces[face].append((u, v))

    faces = ['top', 'right', 'front', 'left', 'back', 'bottom']
    positions = {
        'top': [0, 1],
        'right': [1, 2],
        'front': [1, 1],
        'left': [1, 0],
        'back': [1, 3],
        'bottom': [2, 1]
    }

    fig, axarr = plt.subplots(3, 4, figsize=(10, 8))
    
    for ax in axarr.flatten():
        ax.axis('off')

    for face in faces:
        row, col = positions[face]
        ax = axarr[row, col]
        face_points = cube_map_faces[face]
        if face_points:
            u, v = zip(*face_points)
            ax.scatter(u, v, s=5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            
        ax.grid(False)
        ax.legend().set_visible(False)

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(title)
    plt.show()
    return fig



def plot_surface_no_grid(ax, x, y, z, point=None, title="Superquadric"):
    # Plot the surface
    ax.plot_surface(x, y, z, color='g', alpha=0.5)
    
    # Plot the point and vector, if provided

    ax.set_xlim([-0.65, 0.65])
    ax.set_ylim([-0.65, 0.65])
    ax.set_zlim([-0.65, 0.65])
    
    # Set the title of the plot    
    
    # Hide the grid and axis lines
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))