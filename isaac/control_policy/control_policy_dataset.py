from torch.utils.data import Dataset
import numpy as np

class control_policy_dataset(Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output
    
    def __getitem__(self, item):
        return self.input[item], self.output[item]
    
    def __len__(self):
        return self.input.shape[0]