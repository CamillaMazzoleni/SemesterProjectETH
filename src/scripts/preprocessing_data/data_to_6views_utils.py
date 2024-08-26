
import argparse
from src.utils.json_utils import load_cuboid_data
from src.utils.plotting_utils import load_mesh_data, load_pointcloud_data
import numpy as np

def process_file(file_path, file_extension, output_directory, file_name, visualizer, save_depth=False):
   
    if file_extension == '.json':
        cuboid_data = load_cuboid_data(file_path)
        for idx, component in enumerate(cuboid_data.components):
                visualizer.add_superquadric(
                    f"sq_{idx}",
                    scalings=np.array(component.scale),
                    exponents=np.array([component.epsilon1, component.epsilon2]),
                    translation=np.array(component.position),
                    rotation=np.array(component.rotation),
                    color=np.array(component.color)
                )
    elif file_extension == '.obj':
        #try to load the mesh data from the file
        
        try:
            mesh = load_mesh_data(file_path)
            if mesh is None:
                return
            mesh.compute_vertex_normals()
            #print(f"Loaded mesh with {len(np.asarray(mesh.vertices))} vertices and {len(np.asarray(mesh.triangles))} triangles")
            mesh.translate(-mesh.get_center())
            visualizer.add_geometry(mesh)
            
        except:
            print("Error loading mesh data")
            quit()

    elif file_extension == '.pcd':
        point_cloud = load_pointcloud_data(file_path)
        visualizer.add_geometry(point_cloud)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    visualizer.save_cube_map_screenshots(file_name, output_directory, save_depth=save_depth)


