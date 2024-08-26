import open3d as o3d
import numpy as np
from src.superquadric_difformation.superquadric_class import SuperQuadrics
from src.visualizers.BaseVisualizer import BaseVisualizer
import os
from PIL import Image

class Open3DVisualizer(BaseVisualizer):
    def __init__(self, window_width=800, window_height=800):
        super().__init__(window_width, window_height)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=window_width, height=window_height)
        self.setup_render_options()

    def setup_render_options(self):
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0, 0, 0])  # Set background to white
        opt.show_coordinate_frame = True  # Show coordinate frame
        opt.mesh_show_wireframe = False  # Do not display wireframe on the mesh
        opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color  # Set shading option to color
        opt.point_show_normal = False  # Do not show normals for points
        opt.show_coordinate_frame = False  # Hide coordinate frame
        opt.mesh_show_back_face = True  # Show back faces of the mesh

        # If you have a JSON file with additional options, load it
        render_option_file = "src/visualizers/render_option.json"
        if os.path.exists(render_option_file):
            opt.load_from_json(render_option_file)


    def add_superquadric(self, name: str, scalings: np.array, exponents: np.array,
                         translation: np.array, rotation: np.array, color: np.array, resolution: int = 100):
        exponent_sq = np.array([exponents[0], exponents[1]])
  
        sq = SuperQuadrics(scalings, exponent_sq , resolution)
        vertices = np.stack((sq.x.flatten(), sq.y.flatten(), sq.z.flatten()), axis=-1)
        
        """
        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(111, projection='3d')
        sq.plot(ax, None, "")
        plt.show()
        """


        triangles = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                triangles.append([i * resolution + j, (i + 1) * resolution + j + 1, (i + 1) * resolution + j])
                triangles.append([i * resolution + j, i * resolution + j + 1, (i + 1) * resolution + j + 1])

        mesh_sq = o3d.geometry.TriangleMesh()
        mesh_sq.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_sq.triangles = o3d.utility.Vector3iVector(triangles)
        mesh_sq.compute_vertex_normals()
        
        # Apply color
        #divide the tuple

        color = color / 255.0  # Normalize the color
        mesh_sq.paint_uniform_color(color)

        # Apply translation
        mesh_sq.translate(translation)

        # Apply rotation from 3x3 matriz
        mesh_sq.rotate(rotation)

        self.vis.add_geometry(mesh_sq)   
        

        return mesh_sq, vertices


    def capture_front_picture(self, output_filename: str):
        """
        Captures a screenshot of the current scene from the front view.
        
        Parameters:
        - output_filename: The path where the image will be saved.
        """
        ctr = self.vis.get_view_control()
        
        # Set the front view
        ctr.set_front([0, 0, -1])  # Camera looking towards the negative z-axis
        ctr.set_lookat([0, 0, 0])  # Center of the scene
        ctr.set_up([0, 1, 0])      # The up direction is along the y-axis

        # Poll events and update the renderer
        self.vis.poll_events()
        self.vis.update_renderer()

        try:
            # Capture and save the screenshot
            self.vis.capture_screen_image(output_filename)
            print(f"Front view screenshot saved as {output_filename}")
        except Exception as e:
            print(f"Failed to save front view screenshot: {e}")
    

    def add_complete_cuboid(self, cuboid_data):
        for idx, component in enumerate(cuboid_data.components):
            self.add_superquadric(
                name=f"sq_{idx}",
                scalings=np.array(component.scale),
                exponents=np.array([component.epsilon1, component.epsilon2]),
                translation=np.array(component.position),
                rotation=np.array(component.rotation),
                color=np.array(component.color)
            )
    

    
    def save_views(self, base_filename: str, views: dict, colors: dict, output_folder_path: str, save_camera_params: bool = False, save_depth: bool = False, distance: float = 1.0):
        output = {
        "cuboid_cubemaps": {},
        "camera_parameters": {}
    }
        
        print(save_depth)
    
        for view_name, view_direction in views.items():
            ctr = self.vis.get_view_control()
            ctr.set_front(view_direction)
            ctr.set_lookat([0, 0, 0])
            
            if view_name in ["top", "bottom"]:
                ctr.set_up([0, 0, 1])
            else:
                ctr.set_up([0, 1, 0])

            # Apply the camera extrinsics
            camera_params = ctr.convert_to_pinhole_camera_parameters()
            extrinsic = { 
                "left": np.array([
            [0, 0, 1, 0],    # Rotate to look along the positive x-axis
            [0, 1, 0, 0],    # No rotation around y
            [-1, 0, 0, distance],  # Camera positioned at x = distance
            [0, 0, 0, 1]
                ]),
                "right": np.array([
            [0, 0, -1, 0],   # Rotate to look along the negative x-axis
            [0, 1, 0, 0],    # No rotation around y
            [1, 0, 0, distance],  # Camera positioned at x = distance
            [0, 0, 0, 1]
                ]),
                "top": np.array([
            [1, 0, 0, 0],    # No rotation around x
            [0, 0, 1, 0],    # Rotate to look along the positive z-axis
            [0, -1, 0, distance],  # Camera positioned at y = distance
            [0, 0, 0, 1]
                ]),
                "bottom": np.array([
            [1, 0, 0, 0],    # No rotation around x
            [0, 0, -1, 0],   # Rotate to look along the negative z-axis
            [0, 1, 0, distance],  # Camera positioned at y = distance
            [0, 0, 0, 1]
                ]),
                "front": np.array([
            [1, 0, 0, 0],    # No rotation around x
            [0, 1, 0, 0],    # No rotation around y
            [0, 0, 1, distance],  # Camera positioned at z = distance
            [0, 0, 0, 1]
                ]),
                "back": np.array([
            [1, 0, 0, 0],    # No rotation around x
            [0, 1, 0, 0],    # No rotation around y
            [0, 0, -1, distance],  # Camera positioned at z = distance
            [0, 0, 0, 1]
                ])
            }
            
            camera_params.extrinsic = extrinsic[view_name]
            ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
            # Set the lookat point to the center of the scene
            ctr.set_front(views[view_name]) # Camera looking straight down the z-axis
            if view_name in ["top", "bottom"]:
                ctr.set_up([0, 0, 1])
            else:
                ctr.set_up([0, 1, 0])

            

            self.vis.poll_events()
            self.vis.update_renderer()

            component_name_view = f"{base_filename}_view_{view_name}.png"
            cuboid_filename = os.path.join(output_folder_path, component_name_view)

            if save_depth:
                
                depth_filename = os.path.join(output_folder_path, f"{base_filename}_depth_{view_name}.png")
                self.vis.capture_depth_image(depth_filename)
                depth_image = o3d.io.read_image(depth_filename)
                depth_array = np.asarray(depth_image)
                #create a figure with the same dimesnions as the image
                depth_normalized_image = Image.fromarray((depth_array * 255).astype(np.uint8))
                depth_normalized_image.save(depth_filename)
                print(f"Saved depth image to {depth_filename}")

            else:
                try:
                    self.vis.capture_screen_image(cuboid_filename)
                    print(f"Saved screenshot to {cuboid_filename}") 
                    output["cuboid_cubemaps"][view_name] = cuboid_filename
                except Exception as e:
                    print(f"Failed to save screenshot for view {view_name}: {e}")

            if save_camera_params:
                camera_params = ctr.convert_to_pinhole_camera_parameters()
                output["camera_parameters"][view_name] = {
                    "intrinsic": camera_params.intrinsic.intrinsic_matrix.tolist(),
                    "extrinsic": camera_params.extrinsic.tolist()
                }

            
                

        # Remove the camera_parameters key if save_camera_params is False
        if not save_camera_params:
            del output["camera_parameters"]

        return output

    def save_cube_map_screenshots(self, base_filename: str, output_folder_path: str, save_depth: bool = False):
        views = {
            "top": [0, 1, 0],
            "bottom": [0, -1, 0],
            "left": [1, 0, 0],
            "right": [-1, 0, 0],
            "front": [0, 0, -1],
            "back": [0, 0, 1]
        }

        colors = {
            "top": [1, 0, 0],      # Red
            "bottom": [0, 1, 0],   # Green
            "left": [0, 0, 1],     # Blue
            "right": [1, 1, 0],    # Yellow
            "front": [1, 0, 1],    # Magenta
            "back": [0, 1, 1]      # Cyan
        }

        return self.save_views(base_filename, views, colors, output_folder_path, save_depth=save_depth)
    

    def add_geometry(self, mesh):
        self.vis.add_geometry(mesh)
    
    def add_pointcloud(self, name: str, points: np.ndarray, color: np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(color)
        self.vis.add_geometry(pcd)
        return pcd
        

    def clear_scene(self):
        self.vis.clear_geometries()

    def close(self):
        self.vis.destroy_window()

    def show(self):
        self.vis.run()
        self.vis.destroy_window()
