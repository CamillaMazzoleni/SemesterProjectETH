import os
import random
import json
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn

from src.cuboid_generator.data_loader import shapenet4096
from src.cuboid_generator.network import Network_Whole
import src.cuboid_generator.utils_pytorch as utils_pt

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed) 


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.E_CUDA)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_folder): 
        os.makedirs(args.output_folder)

    # Save checkpoint path information
    with open(os.path.join(args.output_folder, 'checkpoint.txt'), 'w') as f:
        f.write(args.E_ckpt_path + '\n')
        f.write(args.checkpoint)

    # Load hyperparameters
    with open(os.path.join(args.E_ckpt_path, 'hypara.json')) as f:
        hypara = json.load(f) 

    # Create Model
    Network = Network_Whole(hypara).cuda()
    Network.eval()

    # Load Model
    Network.load_state_dict(torch.load(os.path.join(args.E_ckpt_path, args.checkpoint)))
    print('Load model successfully: %s' % (os.path.join(args.E_ckpt_path, args.checkpoint)))

    color = utils_pt.generate_ncolors(hypara['N']['N_num_cubes'])

    hypara['E'] = {}
    hypara['E']['E_shapenet4096'] = args.input_folder  # Use the input folder for the dataset

    # Create Dataset
    batch_size = 32
    infer_test = True
    if infer_test:
        cur_dataset = shapenet4096('test', hypara['E']['E_shapenet4096'], hypara['D']['D_datatype'], True)
        cur_dataloader = DataLoader(cur_dataset, 
                                    batch_size = batch_size,
                                    shuffle=False, 
                                    num_workers=4, 
                                    pin_memory=True)
        infer(args, cur_dataloader, Network, hypara, 'test', batch_size, color)
  

def infer(args, cur_dataloader, Network, hypara, train_val_test, batch_size, color):
    save_path = os.path.join(args.output_folder, train_val_test)
    if not os.path.exists(save_path): 
        os.makedirs(save_path)
    for j, data in enumerate(cur_dataloader, 0):
        with torch.no_grad():
            points, normals, _, _, names = data
            points, normals = points.cuda(), normals.cuda()
            outdict = Network(pc = points)
            utils_pt.save_cubes_json(outdict['scale'], outdict['rotate'],  outdict['pc_assign_mean'], outdict['exist'], save_path, names, color)
            print(j)
    
    print("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--E_CUDA', default=0, type=int, help='Index of CUDA')
    parser.add_argument('--infer_train', default=True, type=bool, help='If infer training set')
    parser.add_argument('--infer_test', default=True, type=bool, help='If infer test set')
    
    parser.add_argument('--input_folder', default='', type=str, help='Path to input dataset folder')
    parser.add_argument('--output_folder', default='', type=str, help='Path to output folder for saving results')
    parser.add_argument('--E_ckpt_path', default='', type=str, help='Experiment checkpoint path')
    parser.add_argument('--checkpoint', default='', type=str, help='Checkpoint name')

    args = parser.parse_args()
    main(args)
