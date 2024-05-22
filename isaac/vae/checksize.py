import numpy as np

data = np.load("/home/cyx/project/visual_servo/isaac/vae/data/desired_feature04:49PM on May 11, 2024.npy", allow_pickle=True)
for i in range(len(data)):
    print(data[i].keys().shape)
    # print(data[i]['hole_feature_points_to_image'].shape)
    
