
from src.visualizers.Open3dVisualizer import Open3DVisualizer
from src.utils.json_utils import load_json
from src.utils.plotting_utils import create_cube_map_2x3
import os
from src.utils.json_utils import load_cuboid_data      
import numpy as np


def main():
    # list all the files in the prcessed dataset directory
    base_path = '/Users/cmazzoleni/Documents/GitHub/CuboidAbstractionViaSeg/data/processed/chair_cuboid_json'
    output_directory = '/Users/cmazzoleni/Documents/GitHub/CuboidAbstractionViaSeg/data/processed/chair_cubemap'
    os.makedirs(output_directory, exist_ok=True)
    visualizer = Open3DVisualizer() 
    save_json = []

    for file in os.listdir(base_path):
        print(f"Processing file: {file}")
        if file.endswith('.json'):
            file_path = os.path.join(base_path, file)
            cuboid_data = load_cuboid_data(file_path)

            print(cuboid_data)
            print ("  ")
            print ("  ")
            print ("  ")
            print ("  ")

            #process each cuboid in cuboid data
            for idx, component in enumerate(cuboid_data.components):
                visualizer.add_superquadric(
                    f"sq_{idx}",
                    scalings=np.array(component.scale),
                    exponents=np.array([component.epsilon1, component.epsilon2]),
                    translation=np.array(component.position),
                    rotation=np.array(component.rotation),
                    color=np.array(component.color)
                )
                
            #save the screenshot of the complete cuboid
            complete_screenshot_path = os.path.join(output_directory, f"{file[:-5]}_complete_object.png")
            #save it from 6 views
            visualizer.save_cube_map_screenshots(file[:-5], output_directory)
            # Clear the scene before processing the next file
        
        visualizer.clear_scene()

    # Close the visualizer once done
    visualizer.close()

if __name__ == "__main__":
    main()

            
            