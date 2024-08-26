import open3d as o3d
import numpy as np
import os
import json
import plyfile
from superquadric_class import SuperQuadrics
from PIL import Image

data = []
base_dataset_path = '/Users/cmazzoleni/Documents/GitHub/CuboidAbstractionViaSeg/dataset_depth_map/chair'

depthmap_views_folder_path = os.path.join(base_dataset_path, 'depthmap_views')
cuboid_views_folder_path = os.path.join(base_dataset_path, 'cuboid_views')

base_input_path = '/Users/cmazzoleni/Documents/GitHub/CuboidAbstractionViaSeg/inputs_dataset/chair'
origin_cuboid_folder_path = os.path.join(base_input_path, 'cuboid_json')
origin_pointcloud_folder_path = os.path.join(base_input_path, 'segmentation_ply')


# Helper function to save JSON
def save_json(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

# Updated function to save camera parameters
def save_camera_params(view_control, save_path):
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    camera_data = {
        "intrinsic": np.asarray(camera_params.intrinsic.intrinsic_matrix).tolist(),
        "extrinsic": np.asarray(camera_params.extrinsic).tolist(),
    }
    save_json(save_path, camera_data)

# Updated save_pc_depth function to save depth images and camera parameters
def save_pc_depth(vis, view_control, component_name, depthmap_views_folder_path, save_camera_params_once=True):
    output = {
        "camera_params": {},
        "depth_images": {}
    }  # Store camera params and depth images for all views

    camera_params_saved = not save_camera_params_once  # Track if camera params are already saved

    
    for view_name, view_direction in views.items():
        # Set camera view direction and orientation
        ctr = vis.get_view_control()
        ctr.set_front(view_direction)
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1] if view_name in ["top", "bottom"] else [0, 1, 0])
        
        vis.poll_events()
        vis.update_renderer()

        # Define filenames for depth image and camera parameters
        depth_filename = os.path.join(depthmap_views_folder_path, f"{component_name}_view_{view_name}.png")
        camera_params_filename = os.path.join(depthmap_views_folder_path, f"{component_name}_view_{view_name}_camera_params.json")
        
        # Save depth image and camera parameters
        vis.capture_depth_image(depth_filename)
        save_camera_params(view_control, camera_params_filename)
        if not camera_params_saved:
            save_camera_params(view_control, camera_params_filename)
            camera_params_saved = True
            output["camera_params"][view_name] = os.path.relpath(camera_params_filename, base_dataset_path)
        
        output["depth_images"][view_name] = depth_filename

    return output



def create_cube_map(views_dict, output_path):
    # Open all images
    top = Image.open(views_dict['depth_images']['top'])
    bottom = Image.open(views_dict['depth_images']['bottom'])
    front = Image.open(views_dict['depth_images']['front'])
    back = Image.open(views_dict['depth_images']['back'])
    left = Image.open(views_dict['depth_images']['left'])
    right = Image.open(views_dict['depth_images']['right'])

    # Convert images to RGB if they are not already
    top = top.convert("RGB")
    bottom = bottom.convert("RGB")
    front = front.convert("RGB")
    back = back.convert("RGB")
    left = left.convert("RGB")
    right = right.convert("RGB")

    width, height = top.size
    
    # Create a new image with the appropriate size (3x width by 2x height)
    cube_map = Image.new('RGB', (3 * width, 2 * height))

    # Paste images into the cube map
    cube_map.paste(top, (0, 0))
    cube_map.paste(back, (width, 0))
    cube_map.paste(left, (2 * width, 0))
    cube_map.paste(front, (0, height))
    cube_map.paste(right, (width, height))
    cube_map.paste(bottom, (2 * width, height))

    # Save the cube map
    
    cube_map.save(output_path)

def create_cube_map2(views_dict, output_path):
    # Open all images
    top = Image.open(views_dict['cuboid_cubemaps']['top'])
    bottom = Image.open(views_dict['cuboid_cubemaps']['bottom'])
    front = Image.open(views_dict['cuboid_cubemaps']['front'])
    back = Image.open(views_dict['cuboid_cubemaps']['back'])
    left = Image.open(views_dict['cuboid_cubemaps']['left'])
    right = Image.open(views_dict['cuboid_cubemaps']['right'])

    # Convert images to RGB if they are not already
    top = top.convert("RGB")
    bottom = bottom.convert("RGB")
    front = front.convert("RGB")
    back = back.convert("RGB")
    left = left.convert("RGB")
    right = right.convert("RGB")

    width, height = top.size
    
    # Create a new image with the appropriate size (3x width by 2x height)
    cube_map = Image.new('RGB', (3 * width, 2 * height))

    # Paste images into the cube map
    cube_map.paste(top, (0, 0))
    cube_map.paste(back, (width, 0))
    cube_map.paste(left, (2 * width, 0))
    cube_map.paste(front, (0, height))
    cube_map.paste(right, (width, height))
    cube_map.paste(bottom, (2 * width, height))

    # Save the cube map
    
    cube_map.save(output_path)


views = {
    "top": [0, 1, 0],
    "bottom": [0, -1, 0],
    "left": [1, 0, 0],
    "right": [-1, 0, 0],
    "front": [0, 0, -1],
    "back": [0, 0, 1]
}

colors = {
    "top": [1, 0, 0],      # Rosso
    "bottom": [0, 1, 0],   # Verde
    "left": [0, 0, 1],     # Blu
    "right": [1, 1, 0],    # Giallo
    "front": [1, 0, 1],    # Magenta
    "back": [0, 1, 1]      # Ciano
}

def add_superquadric(self, name: str, scalings: np.array=np.array([1.0, 1.0, 1.0]),
                         exponents: np.array=np.array([2.0, 2.0, 1.0]), translation: np.array=np.array([0.0, 0.0, 0.0]),
                         rotation: np.array=np.array([1.0, 0.0, 0.0, 0.0]), color: np.array=np.array([255, 255, 255]),
                         resolution: int=100, visible: bool=True):
        """Adds a superquadric mesh to the scene."""

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
        color = color / 255.0  # Normalize the color
        mesh_sq.paint_uniform_color(color)

        # Apply translation
        mesh_sq.translate(translation)

        # Apply rotation from 3x3 matriz
        mesh_sq.rotate(rotation)
        

        return mesh_sq, vertices



def save_pc_cuboid(vis, view_control, component_name, cuboid_views_folder_path):
    output = {
        "cuboid_cubemaps": {}
    }
    for view_name, view_direction in views.items():
        ctr = vis.get_view_control()
        ctr.set_front(view_direction)
        ctr.set_lookat([0, 0, 0])
        if view_name in ["top", "bottom"]:
            ctr.set_up([0, 0, 1])
        else:
            ctr.set_up([0, 1, 0])
        vis.poll_events()
        vis.update_renderer()

        component_name_view = f"{component_name}_view_{view_name}.png" 
      
        
        cuboid_filename = os.path.join(cuboid_views_folder_path, component_name_view)
        vis.capture_screen_image(cuboid_filename)
        output["cuboid_cubemaps"][view_name] = cuboid_filename

    return output




def save_complete_cuboid(vis, view_control, filename):
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    
    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(filename)

     

def main():
    #prepare dataset

    segmentation_list = os.listdir(origin_pointcloud_folder_path)
    new_dataset = []

    # Initialize the visualizer with a specific window size
    window_width = 800
    window_height = 800
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_width, height=window_height)
    
    for segmentation in segmentation_list:
        shapenet_id = segmentation.split('_')[0]
    
        depthmap_views_id_folder_path = os.path.join(depthmap_views_folder_path, shapenet_id)
        cuboid_views_id_folder_path = os.path.join(cuboid_views_folder_path, shapenet_id)
        

        os.makedirs(depthmap_views_id_folder_path, exist_ok=True)
        os.makedirs(cuboid_views_id_folder_path, exist_ok=True)

        segmentation_path = os.path.join(origin_pointcloud_folder_path, segmentation)

        json_file = os.path.join(origin_cuboid_folder_path, f"{shapenet_id}.json")
        with open(json_file, 'r') as file:
            cuboid_data = json.load(file)
        cuboid_colors = list()
        for component in cuboid_data["components"]:
            color = component["color"]
            r, g, b = color
            color = np.array([r, g, b]) / 255.0
            if not any(np.all(color == cuboid_color) for cuboid_color in cuboid_colors):
                cuboid_colors.append(tuple(color))

        plydata = plyfile.PlyData.read(segmentation_path)
        vertex = plydata['vertex']
        points = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T
        colors = np.vstack((vertex['red'], vertex['green'], vertex['blue'])).T / 255.0
        
        point_cloud_colors = set(tuple(color) for color in colors)

        new_item = {
            "shapenet_id": shapenet_id,
            "class_id": "chair",
            "nr_cubes": len(cuboid_colors),
            "complete_cuboid": "",
            "component_origins": [],
            "component_targets": [],
            "depth_view_camera_params": {},
            "cuboid_cubemaps": {},
            "depth_cubemaps": {},
        }

        camera_params_saved = False
        
        for color in point_cloud_colors:
            if color not in cuboid_colors:
                continue
            index = cuboid_colors.index(color)
            mask = np.all(colors == color, axis=1)
            segmented_points = points[mask]
            segmented_points -= np.mean(segmented_points, axis=0)
            segmented_points_pcd = o3d.geometry.PointCloud()
            segmented_points_pcd.points = o3d.utility.Vector3dVector(segmented_points)
            segmented_points_pcd.colors = o3d.utility.Vector3dVector([color] * len(segmented_points))
            segmented_points_pcd.paint_uniform_color(color)

            vis.add_geometry(segmented_points_pcd)

            pointcloud_views = save_pc_depth(vis, vis.get_view_control(), f"component_{index}", depthmap_views_id_folder_path, not camera_params_saved)
            if not camera_params_saved:
                new_item["depth_view_camera_params"] = pointcloud_views["camera_params"]
                camera_params_saved = True
            # Create depth cubemap
            depth_cubemap_filename = os.path.join(depthmap_views_id_folder_path, f"component_{index}_cubemap.jpg")
            create_cube_map(pointcloud_views, depth_cubemap_filename)
            new_item["depth_cubemaps"][f"component_{index}"] = depth_cubemap_filename

            vis.clear_geometries()
            vis.clear_geometries()        



        for i, component in enumerate(cuboid_data["components"]):
            scale = np.array(component["scale"])
            rotation = np.array(component["rotation"])
            position = np.array([0, 0, 0])
            color = np.array(component["color"])
            exponents = np.array([component["epsilon1"], component["epsilon2"]])
            
            mesh_sq, _ = add_superquadric(vis, f"component_{i}", scale, exponents, position, rotation, color, 100, True)

            cubemap_views_component_folder_path = os.path.join(cuboid_views_id_folder_path, f"component_{i}")
            #os.makedirs(cubemap_views_component_folder_path, exist_ok=True)

            vis.add_geometry(mesh_sq)
            opt = vis.get_render_option()
            opt.background_color = np.array([1, 1, 1])  # Set background to black
            opt.show_coordinate_frame = True  # Show coordinate frame
            opt.mesh_show_wireframe = False  # Display wireframe on the mesh
            opt.mesh_shade_option =  o3d.visualization.MeshShadeOption.Color
            opt.point_show_normal = False
            opt.show_coordinate_frame = False   
            opt.mesh_show_back_face = True

            pc_cuboid = save_pc_cuboid(vis, vis.get_view_control(), f"component_{i}", cuboid_views_id_folder_path)

            cuboid_cubemap_filename = os.path.join(cuboid_views_id_folder_path, f"component_{i}_cubemap.jpg")
            

            create_cube_map2(pc_cuboid, cuboid_cubemap_filename)
            new_item["cuboid_cubemaps"][f"component_{i}"]= cuboid_cubemap_filename

            vis.clear_geometries()
            vis.clear_geometries()

        for i, component in enumerate(cuboid_data["components"]):
            scale = np.array(component["scale"])
            rotation = np.array(component["rotation"])
            position = np.array(component["position"])
            color = np.array(component["color"])
            exponents = np.array([component["epsilon1"], component["epsilon2"]])
            
            mesh_sq, _ = add_superquadric(vis, f"component_{i}", scale, exponents, position, rotation, color, 100, True)

            cubemap_views_component_folder_path = os.path.join(cuboid_views_id_folder_path, f"component_{i}")
            #os.makedirs(cubemap_views_component_folder_path, exist_ok=True)

            vis.add_geometry(mesh_sq)
            opt = vis.get_render_option()
            opt.background_color = np.array([1, 1, 1])  # Set background to black
            opt.show_coordinate_frame = True  # Show coordinate frame
            opt.mesh_show_wireframe = False  # Display wireframe on the mesh
            opt.mesh_shade_option =  o3d.visualization.MeshShadeOption.Color
            opt.point_show_normal = False
            opt.show_coordinate_frame = False   
            opt.mesh_show_back_face = True

        compelted_cuboid_filename = os.path.join(base_dataset_path, f"{shapenet_id}_complete_object.png")
        save_complete_cuboid(vis, vis.get_view_control(), compelted_cuboid_filename)
        new_item['complete_cuboid'] = os.path.basename(compelted_cuboid_filename)  # Use relative path
        vis.clear_geometries()
        new_dataset.append(new_item)

    save_json('dataset.json', new_dataset)
        #test code

if __name__ == "__main__":
    main()
    