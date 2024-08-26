import os, sys, h5py
import numpy as np
import torch
import torch.utils.data as data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

class shapenet4096(data.Dataset):
    def __init__(self, phase, data_root, data_type, if_4096):
        super().__init__()
        self.folder = data_type + '/'
        """
        if phase == 'train':
            self.data_list_file = data_root + data_type + '_train.npy' 
        else:
            self.data_list_file = data_root + data_type + '_test.npy'
        self.data_dir = data_root + self.folder
        self.data_list = np.load(self.data_list_file)
        print(self.data_list)
        """

        self.data_dir = data_root + self.folder
        # Load training data
        train_data_list_file = data_root + data_type + '_train.npy'
        #train_data_list = np.load(train_data_list_file)
        
        # Load test data
        #test_data_list_file = data_root + data_type + '_test.npy'
        #self.data_list = np.load(train_data_list_file)
        self.data_list = np.load(train_data_list_file)

       
        
        # Concatenate training and test data
        #self.data_list = np.concatenate((train_data_list, test_data_list), axis=0)
        
    def __getitem__(self, idx):
        cur_name = self.data_list[idx].split('.')[0]
        cur_data = torch.from_numpy(np.load(self.data_dir + self.data_list[idx])).float()
        cur_points = cur_data[:,0:3]
        cur_normals = cur_data[:,3:]
        cur_points_num = 4096
        cur_values = -1
        return cur_points, cur_normals, cur_points_num, cur_values, cur_name
        
    def __len__(self):
        return self.data_list.shape[0]

class ShapeNet(data.Dataset):
    def __init__(self, data_root, data_type):
        super().__init__()
        self.folder = data_type + '/'
        self.data_dir = data_root
        #self.data_list = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')][:5]
        self.data_list = [
            '2f4fe1db48f0cac1db573653825dd010.npz',
            'b53373122c7964c531a0ecc0d5a7b3d5.npz',
            '63d45791c56483bb2662409120160a57.npz',
            '704179dd47a2282e676de9b6e111da8b.npz',
            '2f6b0ddf12d1311795bea7c29e873d16.npz'
        ]
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        cur_name = self.data_list[idx]
        cur_path = os.path.join(self.data_dir, cur_name)

        # Debugging print statements
        print(f"Loading file: {cur_path}")

        # Load the data
        try:
            cur_data = np.load(cur_path)
            print(f"Loaded data keys: {list(cur_data.keys())}")
            
            # Access the correct key
            cur_points_array = cur_data['pointcloud']  # replace 'points' with 'pointcloud'
            print(f"Points shape: {cur_points_array.shape}")

            # Assuming the pointcloud array contains both points and normals in appropriate columns
            cur_points = torch.from_numpy(cur_points_array[:, 0:3]).float()
            cur_normals = torch.from_numpy(cur_points_array[:, 3:6]).float()
            cur_points_num = cur_points.shape[0]
        except KeyError as e:
            print(f"Key error: {e} in file {cur_path}")
            raise e
        except Exception as e:
            print(f"Error loading {cur_path}: {e}")
            raise e

        cur_values = -1
        return cur_points, cur_normals, cur_points_num, cur_values, cur_name.split('.')[0]




