from src.scripts.preprocessing_data.data_to_6views_utils import process_file
import os
from src.visualizers.Open3dVisualizer import Open3DVisualizer
from src.utils.dataset_utils import save_to_txt, read_from_txt

def main():
    input_folder = "/Users/cmazzoleni/Documents/GitHub/CuboidAbstractionViaSeg/data/raw/03001627"
    output_folder = "/Users/cmazzoleni/Documents/GitHub/CuboidAbstractionViaSeg/data/processed/chair_mesh_6views_depth"
    #for folder in folder
    visualizer = Open3DVisualizer()
    
    output_file = "output.txt"
    save_to_txt(input_folder, output_file)

    folders = read_from_txt(output_file)
    
    # Define the range you want to process (e.g., from 80 to 160)
    start_index = 4029
    end_index = 4500
    folders_to_process = folders[start_index:end_index]

    for i, folder in enumerate(folders_to_process, start=start_index):
        model_folder = os.path.join(input_folder, folder, 'models')
        
        if not os.path.exists(model_folder):
            print(f"Skipping {folder} because 'models' folder does not exist.")
            continue

        print(f"Processing folder {i + 1}: {model_folder}")

        for file in os.listdir(model_folder):
            if file.endswith('.obj'):
                file_path = os.path.join(model_folder, file)
                print(f"Processing {file_path}...") 
                

                process_file(file_path, '.obj', output_folder, folder, visualizer, save_depth=True)
            
                visualizer.clear_scene()    

if __name__ == "__main__":
    main()