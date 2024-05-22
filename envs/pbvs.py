import numpy as np
from scipy.spatial.transform import Rotation as R
class PBVSNumpy:
    def __init__(self, T_w_dc=np.zeros((4,4)),T_w_c=np.zeros((4,4))):
        self.T_w_dc = T_w_dc
        self.T_w_c = T_w_c
        self.vel = np.ones((6,))

    def cal_action(self, T_w_dc, T_w_c):
        T_w_dc = T_w_dc
        self.T_w_c = T_w_c
        self.T_dc_c = np.linalg.inv(self.T_w_dc).dot(self.T_w_c)
        RT = self.T_dc_c[:3,:3].T
        t_dc_c = self.T_dc_c[:3,3]
        thetau = R.from_matrix(self.T_dc_c[:3, :3]).as_rotvec()
        self.vel = np.ones((6,))
        self.vel[:3] = -0.5 * np.dot(RT, t_dc_c)
        self.vel[3:] = -0.5 * thetau
        return self.vel

    