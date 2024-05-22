from torch.utils.data import Dataset
import numpy as np
import os

class vae_dataset(Dataset):
    def __init__(self, root_dir, file_name):
        self.root_dir = root_dir
        self.data = np.load(os.path.join(root_dir, file_name), allow_pickle=True)
    
    def __getitem__(self, item):
        data_dict = self.data[item]
        hole_feature_points_to_image = data_dict['hole_feature_points_to_image']
        peg_feature_points_to_image = data_dict['peg_feature_points_to_image']
        # normalize the data

        hole_feature_points_to_image = data_dict['hole_feature_points_to_image']
        peg_feature_points_to_image = data_dict['peg_feature_points_to_image']
        return np.append(hole_feature_points_to_image, peg_feature_points_to_image, axis=0)
    
    def __len__(self):
        return len(self.data)