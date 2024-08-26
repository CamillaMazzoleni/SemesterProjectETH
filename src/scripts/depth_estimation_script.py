import os
import argparse
from src.depth_processing.DepthEstimator import DepthEstimator
from PIL import Image

def ensure_output_folder(output_folder):
    """Ensure the output folder exists and is writable."""
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        except Exception as e:
            print(f"Failed to create output folder: {output_folder}. Error: {e}")
            exit(1)
    elif not os.access(output_folder, os.W_OK):
        print(f"Output folder is not writable: {output_folder}")
        exit(1)
    else:
        print(f"Using existing output folder: {output_folder}")

def process_image(depth_estimator, image_path, args):
    """Process a single image for depth estimation."""
    depth_map = depth_estimator.estimate_depth(image_path)

    # Generate a file name from the image file name and append '_depth'
    original_name = os.path.splitext(os.path.basename(image_path))[0]
    file_name = f"{original_name}_depth"

    # Visualize the depth map
    if args.visualize:
        save_path = os.path.join(args.output_folder, f"{file_name}_visualized.png") if args.save_output else None
        depth_estimator.visualize_depth(depth_map, save_path=save_path)

    # Save the depth map
    if args.save_output:
        ensure_output_folder(args.output_folder)
        output_folder = args.output_folder if args.output_folder else os.path.dirname(image_path)
        depth_estimator.save_depth(depth_map, output_folder, file_name=file_name)

def main(args):
    # Initialize the depth estimator with the selected model
    depth_estimator = DepthEstimator(model_type=args.model_type, device=args.device)

    if os.path.isdir(args.image_path):
        # Process all image files in the directory
        for filename in os.listdir(args.image_path):
            file_path = os.path.join(args.image_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(f"Processing {file_path}...")
                process_image(depth_estimator, file_path, args)
    else:
        # Process a single image
        process_image(depth_estimator, args.image_path, args)

    # Comparison functionality can be added later or integrated as needed
    if args.compare_with:
        comparison_depth_map = depth_estimator.estimate_depth(args.compare_with)
        depth_estimator.compare_depths(depth_map, comparison_depth_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Estimation Script")

    parser.add_argument("--model_type", type=str, default="depth-anything",
                        choices=["depth-anything", "marigold"],
                        help="Model type to use for depth estimation.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run the model on.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image or directory containing images.")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the depth map.")
    parser.add_argument("--save_output", action="store_true",
                        help="Save the depth map to an output folder.")
    parser.add_argument("--output_folder", type=str,
                        help="Folder to save the output depth map.")
    parser.add_argument("--file_name", type=str, default="depth",
                        help="File name for saving the depth map (ignored if processing a directory).")
    parser.add_argument("--save_path", type=str,
                        help="Path to save the visualized depth map.")
    parser.add_argument("--compare_with", type=str,
                        help="Path to another image to compare depth maps.")

    args = parser.parse_args()
    main(args)
