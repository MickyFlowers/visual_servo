import numpy as np
from scipy.spatial.transform import Rotation as R

class PBVSNumpy:
    def __init__(self, T_w_dc=np.zeros((4,4)),T_w_c=np.zeros((4,4))):
        self.T_w_dc = T_w_dc
        self.T_w_c = T_w_c
        self.vel = np.ones((6,))

    def cal_action(self, T_w_dc, T_w_c):
        self.T_w_dc = T_w_dc
        self.T_w_c = T_w_c
        self.T_dc_c = np.linalg.inv(self.T_w_dc).dot(self.T_w_c)
        RT = self.T_dc_c[:3,:3].T
        t_dc_c = self.T_dc_c[:3,3]
        thetau = R.from_matrix(self.T_dc_c[:3, :3]).as_rotvec()
        self.vel = np.ones((6,))
        self.vel[:3] = -0.5 * np.dot(RT, t_dc_c)
        self.vel[3:] = -0.5 * thetau
        return self.vel

    def cal_action_curve(self,cur_wcT, tar_wcT, points):
        wPo = np.mean(points, axis=0)  # w: world frame, o: center, P: points
        tar_cwT = np.linalg.inv(tar_wcT)
        tPo = wPo @ tar_cwT[:3, :3].T + tar_cwT[:3, 3]
        # t: target camera frame, c: current camera frame, w: world frame

        cur_cwT = np.linalg.inv(cur_wcT)
        cPo = wPo @ cur_cwT[:3, :3].T + cur_cwT[:3, 3]

        tcT = tar_cwT @ cur_wcT
        u = R.from_matrix(tcT[:3, :3]).as_rotvec()

        v = -(tPo - cPo + np.cross(cPo, u))
        w = -u
        vel = np.concatenate([v, w])

        tPo_norm = np.linalg.norm(tPo)
        vel_si = np.concatenate([v / (tPo_norm + 1e-7), w])

        return vel, (tPo_norm, vel_si)