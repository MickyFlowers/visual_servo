from omni.isaac.core.utils.rotations import quat_to_rot_matrix
import numpy as np
from scipy.spatial.transform import Rotation as R

def calc_trans_matrix(pos, ori):
    matrix = np.eye(4)
    matrix[:3, 3] = pos
    matrix[:3, :3] = quat_to_rot_matrix(ori)
    return matrix

def transform_points(points, trans_matrix):
    points = points.reshape(-1, 3)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    return np.matmul(trans_matrix, points.T).T[:, :3].reshape(-1, 3)

def calc_pose_from_vel(trans_matrix, vel, dt):
    dT = np.eye(4)
    dT[:3, :3] = R.from_rotvec(vel[3:] * dt).as_matrix()
    dT[:3, 3] = vel[:3] * dt
    return trans_matrix @ dT