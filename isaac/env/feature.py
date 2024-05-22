import numpy as np
from utils.rotations import calc_trans_matrix
class feature:
    def __init__(self, hole_points, peg_points, camera_intrinsics=None):
        self.hole_points = hole_points
        self.peg_points = peg_points
        if camera_intrinsics == None:
            self.camera_intrinsics = np.array([[616.56402588,   0.        , 330.48983765],
                                                [  0.        , 616.59606934, 233.84162903],
                                                [  0.        ,   0.        ,   1.        ]])
        else:
            self.camera_intrinsics = camera_intrinsics


    
    def project_points_to_img(self, camera_trans_matrix, hole_trans_matrix, peg_trans_matrix):

        # transform hole points to camera frame
        hole_to_camera_matrix = np.linalg.inv(camera_trans_matrix) @ hole_trans_matrix
        hole_points_in_camera_frame = self.transform_points(self.hole_points, hole_to_camera_matrix)

        # transform peg points to camera frame
        peg_to_camera_matrix = np.linalg.inv(camera_trans_matrix) @ peg_trans_matrix
        peg_points_in_camera_frame = self.transform_points(self.peg_points, peg_to_camera_matrix)
        # print(peg_points_in_camera_frame)

        # project hole points to image
        hole_points_in_img = self.project_points(hole_points_in_camera_frame)
        peg_points_in_img = self.project_points(peg_points_in_camera_frame)

        return hole_points_in_img, peg_points_in_img

    
    def transform_points(self, points, trans_matrix):
        points = points.reshape(-1, 3)
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        return np.matmul(trans_matrix, points.T).T[:, :3].reshape(-1, 3)
    
    def project_points(self, points):
        
        points = points.reshape(-1, 3)
        points = np.matmul(self.camera_intrinsics, points.T).T
        points = points[:, :2] / points[:, 2:]
        return points