import torch
import torch.nn as nn
import torch.nn.functional as F


import copy
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import datasets, models, transforms
from scipy.spatial.transform import Rotation as R
from util import ekf
import copy
from util.cor_transform import v2m,m2v

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

mtx = np.load('/home/shahao/yhx/simple-pybullet-ur5/npy/color_intr.npy')
dist = np.load('/home/shahao/yhx/simple-pybullet-ur5/npy/dist.npy')
# objPoints = np.load('/home/shahao/yhx/simple-pybullet-ur5/npy/pt_apriltag.npy')[:, :3]
objPoints = np.load('/home/shahao/yhx/simple-pybullet-ur5/npy/pt_apriltag_augori.npy')[:, :3]
# objPoints = np.load('/home/shahao/yhx/simple-pybullet-ur5/npy/pt_horse.npy')[:, :3]
objPoints = np.float32(objPoints)

colors = np.array([
[255, 0, 0],   # orange
[0, 0, 255],   # orange
[0, 255, 0],  # purpo
[18, 153, 255],  # green
[0, 178, 230],  #深蓝色 DenseNet
[158, 218, 229],  #灰青色 AE
[192, 192, 192],   #灰色
[119, 224, 219],  # 淡青色 HPN-nodagger
[158, 218, 229],  #灰青色 AE
[98, 206, 236],  # 灰蓝色
[40, 170, 170]]) # 浅紫色
# colors = colors / 255

import argparse
from TD3.lib.net.scnet_integration_640 import SCNetIntegration as scnnetIntegration
from TD3.lib.net.resnet_integration_640 import SCNetIntegration as resnetIntegration
from TD3.lib.dataset.align_data import Image

def image_preprocess(bgr):
    rgb_img = Image(bgr)
    # rgb_img.central_crop()
    # rgb_img.resize((384, 384))
    # rgb_img.resize((240, 320))
    # rgb_img.resize((480, 640))
    # rgb_img.rotate(np.pi, (240, 320))
    rgb_img.normalise()
    rgb_img.img = rgb_img.img.transpose((2, 0, 1))
    return rgb_img.img

def get_args():
    parser = argparse.ArgumentParser(description='Train the SCNet on images and target landmarks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', dest='epochs', metavar='E', type=int, default=1000, help='Number of epochs')
    parser.add_argument('-de', '--decay-epochs', dest='decay_epochs', metavar='DE', type=int, default=500, help='Number of decay epochs')
    parser.add_argument('-b', '--batch-size', dest='batchsize', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('-l', '--learning-rate', dest='lr', metavar='LR', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.2, help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir', type=str, default="./data/valid_data0809/",
                        help='Path of dataset for training and validation')
    parser.add_argument('-a', '--assis-dir', dest='assis_dir', type=str, default="/home/hongxiang/CoRL2021/caps/CAPS-MegaDepth-release-light/train/",
                        help='Path of dataset for training and validation')
    parser.add_argument('-m', '--model-dir', dest='model_dir', type=str, default="./checkpoints/",
                        help='Path of trained model for saving')

    return parser.parse_args()

class RES_horse:
    def __init__(self):
        self.device = device
        self.args = get_args()
        self.net = resnetIntegration(args=self.args, device=self.device, input_channel=3, num_labels=4, learning_rate=self.args.lr)
        self.net.to(device=self.device, dtype=torch.float32)
        # cp = '/home/shahao/yhx/IBVS_hw/fpointlearning/SCN_pytorch/checkpoints/fp_learning_210817_0141_100.pth'
        # cp = '/home/shahao/yhx/IBVS_hw/fpointlearning/SCN_pytorch/checkpoints/fp_learning_210821_0155_235.pth'
        # cp = '/home/shahao/yhx/simple-pybullet-ur5-modeldatesetlog/models/horse/TD3_vs_gym:charge-v1_10_230110_0020_horse_real_ob100_train5_obs_resnet_6300_observer'
        cp = '/home/shahao/yhx/simple-pybullet-ur5-modeldatesetlog/models/TD3_vs_gym:charge-v1_10_230116_1951_horse_real_ob100_train5_coe_resnet_dagger3/TD3_vs_gym:charge-v1_10_230116_1951_horse_real_ob100_train5_coe_resnet_dagger3_770_observer'
        self.net.load_state_dict(torch.load(cp, map_location=self.device),strict=False)

    def vis(self,bgr,ideal_corners,cam_obs_goal):
        # image_ = bgr.copy()
        image_ = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        pt_num = ideal_corners.shape[0]
        for i in range(pt_num):
            cv2.circle(image_, (int(cam_obs_goal[i, 0]), int(cam_obs_goal[i, 1])), 2, (int(colors[i,0]),int(colors[i,1]),int(colors[i,2])), 5)  # 画圆
            cv2.circle(image_, (int(ideal_corners[i, 0]), int(ideal_corners[i, 1])), 2,(int(colors[i,0]),int(colors[i,1]),int(colors[i,2])), 5)  # 画圆
        cv2.imshow('image_',image_)
        cv2.waitKey(1)

    def predict(self,bgr):
        with torch.no_grad():
            # bgr = self.d435_stream.get()
            # ori_iamge = bgr.copy()
            bgr_ori = bgr.copy()
            bgr = image_preprocess(bgr)
            image = torch.FloatTensor(bgr).unsqueeze(0).to(self.device)
            self.net.input_image = image
            e_xy, v_xy, prediction = self.net.pred()
            landmarks = e_xy.cpu().numpy()
            return landmarks,bgr_ori

    def process(self,bgr):
        # with torch.no_grad():
            # bgr = image_preprocess(bgr)
            # image = torch.FloatTensor(bgr).unsqueeze(0).to(self.device)
        self.net.input_image = bgr
        e_xy, v_xy, prediction = self.net.pred()
        return e_xy.view(-1,5,2)

class RES:
    def __init__(self):
        self.device = device
        self.args = get_args()
        self.net = resnetIntegration(args=self.args, device=self.device, input_channel=3, num_labels=5, learning_rate=self.args.lr)
        self.net.to(device=self.device, dtype=torch.float32)
        # cp = '/home/hongxiang/CoRL2021/IBVS_hw/fpointlearning/SCN_pytorch/checkpoints/fp_learning_210817_0141_100.pth'
        # cp = '/home/hongxiang/CoRL2021/IBVS_hw/fpointlearning/SCN_pytorch/checkpoints/fp_learning_210821_0155_235.pth'
        # cp = '/home/hongxiang/CoRL2021/fpointlearning/SCN-pytorch/checkpoints/fp_learning_210903_1637_valid_mix0903_45.pth'
        # self.net.load_state_dict(torch.load(cp, map_location=self.device),strict=False)

    def vis(self,bgr,ideal_corners,cam_obs_goal=np.array([[263.5138, 204.3294],
                                [333.6181, 202.5389],
                                [404.5112, 202.8710],
                                [298.0447, 265.4165],
                                [368.3883, 265.1016]])):
        image_ = bgr.copy()
        for i in range(5):
            cv2.circle(image_, (int(cam_obs_goal[i, 0]), int(cam_obs_goal[i, 1])), 2, (0, 0, 255), 2)  # 画圆
        for i in range(5):
            cv2.circle(image_, (int(ideal_corners[i, 0]), int(ideal_corners[i, 1])), 2, (0, 255, 0), 2)  # 画圆
        cv2.imshow('image_',image_)
        cv2.waitKey(1)

    def predict(self,bgr):
        with torch.no_grad():
            # bgr = self.d435_stream.get()
            # ori_iamge = bgr.copy()
            bgr_ori = bgr.copy()
            bgr = image_preprocess(bgr)
            image = torch.FloatTensor(bgr).unsqueeze(0).to(self.device)
            self.net.input_image = image
            e_xy, v_xy, prediction = self.net.pred()
            landmarks = e_xy.cpu().numpy()
            return landmarks,bgr_ori

    def process(self,bgr):
        # with torch.no_grad():
            # bgr = image_preprocess(bgr)
            # image = torch.FloatTensor(bgr).unsqueeze(0).to(self.device)
        self.net.input_image = bgr
        e_xy, v_xy, prediction = self.net.pred()
        return e_xy.view(-1,5,2)

class Apriltag(object):
    def __init__(
        self,
    ):
        self.state_points = None

    def predict(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)
        # print(corners)
        if len(corners) == 0:
            return np.float32(np.array([[211., 367.],
                                 [418., 357.],
                                 [437., 146.],
                                 [193., 129.]])), False
        corners = corners[0][0]
        state_points = corners[[2, 1, 0, 3], :]
        return state_points, True

    def vis(self,bgr,ideal_corners,cam_obs_goal):
        # image_ = bgr.copy()
        image_ = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        pt_num = ideal_corners.shape[0]
        for i in range(pt_num):
            cv2.circle(image_, (int(cam_obs_goal[i, 0]), int(cam_obs_goal[i, 1])), 2, (int(colors[i,0]),int(colors[i,1]),int(colors[i,2])), 5)  # 画圆
            cv2.circle(image_, (int(ideal_corners[i, 0]), int(ideal_corners[i, 1])), 2,(int(colors[i,0]),int(colors[i,1]),int(colors[i,2])), 5)  # 画圆
        cv2.imshow('image_',image_)
        cv2.waitKey(1)
        
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

    def cal_action_curve(self,cur_wcT, tar_wcT, wP=np.mean(objPoints,axis=0).reshape(1,3)):
        wPo = np.mean(wP, axis=0)  # w: world frame, o: center, P: points
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

    def cal_action_curve_pnp(self,state_points, goal_points, mtx=None, wP=np.mean(objPoints,axis=0).reshape(1,3)):
        retval, rvec_, tvec_ = cv2.solvePnP(objPoints, goal_points, mtx, dist,flags=cv2.SOLVEPNP_AP3P)
        goal_rv = np.vstack([tvec_, rvec_]).squeeze(1)
        tar_cwT = v2m(goal_rv)
        # print(f"T_dc_o:{T_dc_o}")

        retval, rvec_, tvec_ = cv2.solvePnP(objPoints[:, :], state_points[:, :], mtx, dist,flags=cv2.SOLVEPNP_AP3P)
        state_rv = np.vstack([tvec_, rvec_]).squeeze(1)
        cur_cwT = v2m(state_rv)

        wPo = np.mean(wP, axis=0)  # w: world frame, o: center, P: points
        # tar_cwT = np.linalg.inv(tar_wcT)
        tPo = wPo @ tar_cwT[:3, :3].T + tar_cwT[:3, 3]
        # t: target camera frame, c: current camera frame, w: world frame

        # cur_cwT = np.linalg.inv(cur_wcT)
        cur_wcT = np.linalg.inv(cur_cwT)
        cPo = wPo @ cur_cwT[:3, :3].T + cur_cwT[:3, 3]

        tcT = tar_cwT @ cur_wcT
        u = R.from_matrix(tcT[:3, :3]).as_rotvec()

        v = -(tPo - cPo + np.cross(cPo, u))
        w = -u
        vel = np.concatenate([v, w])

        tPo_norm = np.linalg.norm(tPo)
        vel_si = np.concatenate([v / (tPo_norm + 1e-7), w])

        return vel, (tPo_norm, vel_si)

    def cal_action_PnP(self, goal_points, state_points):
        # print(objPoints)
        # print(goal_points)

        retval, rvec_, tvec_ = cv2.solvePnP(objPoints, goal_points, mtx, dist)
        goal_rv = np.vstack([tvec_, rvec_]).squeeze(1)
        T_dc_o = v2m(goal_rv)
        # print(f"T_dc_o:{T_dc_o}")

        retval, rvec_, tvec_ = cv2.solvePnP(objPoints[:, :], state_points[:, :], mtx, dist)
        state_rv = np.vstack([tvec_, rvec_]).squeeze(1)
        T_c_o = v2m(state_rv)
        # print(f"T_c_o:{T_c_o}")

        # T_dc_o_ = np.array([[1., 0, 0, 0],
        #                    [0, -1., 0, 0],
        #                    [0, 0, -1., 0.15],
        #                    [0, 0, 0, 1.]])
        # print(T_dc_o.dot(np.linalg.inv(T_c_o)))
        # print(T_dc_o_.dot(np.linalg.inv(T_c_o)))

        # self.T_dc_c = T_dc_o_.dot(np.linalg.inv(T_c_o))
        self.T_dc_c = T_dc_o.dot(np.linalg.inv(T_c_o))
        RT = self.T_dc_c[:3,:3].T
        t_dc_c = self.T_dc_c[:3,3]
        thetau = R.from_matrix(self.T_dc_c[:3, :3]).as_rotvec()
        self.vel = np.ones((6,))
        self.vel[:3] = -0.5 * np.dot(RT, t_dc_c)
        self.vel[3:] = -0.5 * thetau
        return self.vel


class IBVSNumpy:
    def __init__(self, feature_points=5, is_Le = False):
        self.feature_points = feature_points
        self.L = np.ones((feature_points * 2, 6))
        self.pL = np.linalg.pinv(self.L)
        self.goal = np.ones((feature_points, 2))
        self.state = np.ones((feature_points, 2))
        self.error = self.state.reshape(feature_points * 2, ) \
                     - self.goal.reshape(feature_points * 2, )
        self.is_Le = is_Le

    def updateL(self):
        Z = 0.2
        f = 616
        for i in range(self.feature_points):
            x = self.state[i, 0]
            y = self.state[i, 1]
            # Z = self.goal_distance[i]
            self.L[i * 2:i * 2 + 2, :] = np.array(
                [[-f / Z, 0, x / Z, x * y / f, -(f * f + x * x) / f, y],
                 [0, -f / Z, y / Z, (f * f + y * y) / f, -x * y / f, -x]])
        # print('Cond L:{}'.format(np.linalg.cond(self.L)))
        self.pL = np.linalg.pinv(self.L)
        self.error = self.state.reshape(self.feature_points * 2, ) \
                     - self.goal.reshape(self.feature_points * 2, )
        err_x = np.mean(abs(self.error[::2]))
        err_y = np.mean(abs(self.error[1::2]))
        # print('err_x:{}'.format(err_x))
        # print('err_y:{}'.format(err_y))

    def updateLe(self):
        Z = 0.2
        f = 616
        for i in range(self.feature_points):
            x = self.goal[i, 0]
            y = self.goal[i, 1]
            # Z = self.goal_distance[i]
            self.L[i * 2:i * 2 + 2, :] = np.array(
                [[-f / Z, 0, x / Z, x * y / f, -(f * f + x * x) / f, y],
                 [0, -f / Z, y / Z, (f * f + y * y) / f, -x * y / f, -x]])

        self.pL = np.linalg.pinv(self.L)
        self.error = self.state.reshape(self.feature_points * 2, ) \
                     - self.goal.reshape(self.feature_points * 2, )

    def cal_action(self, goal_points, state_points):
        self.state = state_points #* np.array([640.,480.])
        self.goal = goal_points #* np.array([640.,480.])
        if self.is_Le:
            self.updateLe()
        else:
            self.updateL()
        vel = -0.5 * np.dot(self.pL, self.error)
        return vel

class ExtIntAE(nn.Module):
    def __init__(self, state_dim, hidden_layer_size):
        super(ExtIntAE, self).__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, hidden_layer_size)
        self.l3 = nn.Linear(hidden_layer_size,128)
        self.l4 = nn.Linear(128, state_dim)


    def forward(self, state, vis=False, tb=None, step=None):
        bsz = state.shape[0]
        input = state.view(bsz,-1,)
        a = F.relu(self.l1(input))
        hidden = F.relu(self.l2(a))
        a = F.relu(self.l3(hidden))
        return hidden,torch.tanh(self.l4(a))

class ExtIntAE2(nn.Module):
    def __init__(self, state_dim, hidden_layer_size):
        super(ExtIntAE2, self).__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, hidden_layer_size)
        self.l3 = nn.Linear(hidden_layer_size,128)
        self.l4 = nn.Linear(128, state_dim)

    def forward(self, state, vis=False, tb=None, step=None):
        bsz = state.shape[0]
        input = state.view(bsz,-1,)
        a = F.relu(self.l1(input))
        hidden = torch.tanh(self.l2(a))
        a = F.relu(self.l3(hidden))
        return hidden,torch.tanh(self.l4(a))

class BaseController(nn.Module):
    def __init__(self):
        super(BaseController, self).__init__()
        self.l1 = nn.Linear(8, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 6)

    def forward(self, state):
        bsz = state.shape[0]
        input = state.view(bsz,-1,)
        a = F.relu(self.l1(input))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))

class TransferModule(nn.Module):
    def __init__(self):
        super(TransferModule, self).__init__()
        self.l1 = nn.Linear(8+6, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 6)

    def forward(self, state):
        bsz = state.shape[0]
        input = state.view(bsz,-1,)
        a = F.relu(self.l1(input))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))

class TIE_Transfer(object):
    def __init__(
        self,
        max_action
    ):
        self.basecontroller = BaseController().to(device=device, dtype=torch.float32)
        self.encoder = ExtIntAE(7,8).to(device=device, dtype=torch.float32)
        self.transfermodule = TransferModule().to(device=device, dtype=torch.float32)

        self.max_action = max_action
        self.criterion = nn.MSELoss()

        self.actor_optimizer = torch.optim.Adam([
                                                {'params': self.transfermodule.parameters(),'lr': 3e-4}
                                                 ])
    def load_basecontroller(self, filename):
        self.basecontroller.load_state_dict(torch.load(filename + "_actor"),strict=False)

    def load_encoder(self, filename):
        self.encoder.load_state_dict(torch.load(filename),strict=False)

    def load_transfermodule(self, filename):
        self.transfermodule.load_state_dict(torch.load(filename + "_actor"),strict=False)

    def select_action_with_basecontroller(self, errstate):
        errstate = torch.Tensor(errstate).to(device=device, dtype=torch.float32).unsqueeze(0)
        return self.max_action * self.basecontroller(errstate).cpu().data.numpy().flatten()

    def select_action_with_transfermodule(self, errstate,extr,intr):
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f,cx,cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1,1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1,1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1,1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1,1)
        env_parameters = torch.cat([rv, theta, f, cx, cy], dim=1)
        env_latent, recon_env_parameters = self.encoder(env_parameters)

        errstate = torch.Tensor(errstate).to(device=device, dtype=torch.float32).unsqueeze(0)
        cc_vel = self.basecontroller(errstate)

        input = torch.cat([cc_vel, env_latent], dim=1)
        output = self.transfermodule(input)
        return self.max_action * output.cpu().data.numpy().flatten()

    def train(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([rv, theta, f, cx, cy], dim=1)
            env_latent, recon_env_parameters = self.encoder(env_parameters)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            seq_state = batch[7].to(device=device, dtype=torch.float32)
            seq_goal = batch[8].to(device=device, dtype=torch.float32)

            state = seq_state[:,  :, :] #+ state_shift.view(-1, 5, 2)
            goal = seq_goal[:,  :, :] #+ goal_shift.view(-1, 5, 2)
            errstate = (state - goal)/80

            cc_vel = self.basecontroller(errstate)
            input = torch.cat([cc_vel, env_latent], dim=1)
            output = self.max_action * self.transfermodule(input)
            # Compute actor losse
            actor_loss = self.criterion(action_pbvs,output)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def save(self, filename):
        torch.save(self.transfermodule.state_dict(), filename)

class SingleModule(nn.Module):
    def __init__(self):
        super(SingleModule, self).__init__()
        self.l1 = nn.Linear(8+8+8, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 6)

    def forward(self, state):
        bsz = state.shape[0]
        input = state.view(bsz,-1,)
        a = F.relu(self.l1(input))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return torch.tanh(self.l4(a))

class TIE_Transfer2(object):
    def __init__(
        self,
        max_action
    ):
        self.basecontroller = BaseController().to(device=device, dtype=torch.float32)
        # self.encoder = ExtIntAE(7,8).to(device=device, dtype=torch.float32)
        self.encoder = ExtIntAE2(7,8).to(device=device, dtype=torch.float32)
        self.transfermodule = SingleModule().to(device=device, dtype=torch.float32)

        self.max_action = max_action
        self.criterion = nn.MSELoss()

        self.actor_optimizer = torch.optim.Adam([
                                                {'params': self.transfermodule.parameters(),'lr': 3e-4}
                                                 ])

    def load_basecontroller(self, filename):
        self.basecontroller.load_state_dict(torch.load(filename + "_actor"),strict=False)

    def load_encoder(self, filename):
        self.encoder.load_state_dict(torch.load(filename),strict=False)

    def load_transfermodule(self, filename):
        self.transfermodule.load_state_dict(torch.load(filename),strict=False)

    def select_action_with_basecontroller(self, errstate):
        errstate = torch.Tensor(errstate).to(device=device, dtype=torch.float32).unsqueeze(0)
        return self.max_action * self.basecontroller(errstate).cpu().data.numpy().flatten()

    def select_action_with_transfermodule(self, cam_obs_cur,cam_obs_goal,extr,intr):
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f,cx,cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1,1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1,1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1,1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1,1)
        env_parameters = torch.cat([rv, theta, f, cx, cy], dim=1)
        env_latent, recon_env_parameters = self.encoder(env_parameters)

        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1,8) /320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1,8)/320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
        output = self.transfermodule(input)
        return self.max_action * output.cpu().data.numpy().flatten()

    def train(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([rv, theta, f, cx, cy], dim=1)
            env_latent, recon_env_parameters = self.encoder(env_parameters)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(-1,8)/320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(-1,8)/320 - 1

            input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
            output = self.max_action * self.transfermodule(input)
            # Compute actor losse
            actor_loss = self.criterion(action_pbvs,output)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def save(self, filename):
        torch.save(self.transfermodule.state_dict(), filename)

class AdaptionModule(nn.Module):
    def __init__(self):
        super(AdaptionModule, self).__init__()
        self.conv1 = torch.nn.Conv1d(8+8+6, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv1d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(40, 8)

    def forward(self, state):
        bsz = state.shape[0]
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(bsz, -1)
        x = self.fc(x)
        return torch.tanh(x)

class TIE_Transfer3(object):
    def __init__(
        self,
        max_action,
            extr,
            intr
    ):
        self.basecontroller = BaseController().to(device=device, dtype=torch.float32)
        self.encoder = ExtIntAE(7,8).to(device=device, dtype=torch.float32)
        self.transfermodule = SingleModule().to(device=device, dtype=torch.float32)

        self.max_action = max_action
        self.criterion = nn.MSELoss()

        self.transfer_optimizer = torch.optim.Adam([
                                                {'params': self.transfermodule.parameters(),'lr': 3e-4}
                                                 ])

        self.env_latent = 0
        self.initial_env_latent(extr,intr)
        self.adaptionmodule = AdaptionModule().to(device=device, dtype=torch.float32)
        self.adaption_optimizer = torch.optim.Adam([
                                                {'params': self.adaptionmodule.parameters(),'lr': 3e-4}
                                                 ])

    def load_basecontroller(self, filename):
        self.basecontroller.load_state_dict(torch.load(filename + "_actor"),strict=False)

    def load_encoder(self, filename):
        self.encoder.load_state_dict(torch.load(filename),strict=False)

    def load_transfermodule(self, filename):
        self.transfermodule.load_state_dict(torch.load(filename),strict=False)

    def select_action_with_basecontroller(self, errstate):
        errstate = torch.Tensor(errstate).to(device=device, dtype=torch.float32).unsqueeze(0)
        return self.max_action * self.basecontroller(errstate).cpu().data.numpy().flatten()

    def select_action_with_transfermodule(self, cam_obs_cur,cam_obs_goal,extr,intr):
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f,cx,cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1,1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1,1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1,1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1,1)
        env_parameters = torch.cat([rv, theta, f, cx, cy], dim=1)
        env_latent, recon_env_parameters = self.encoder(env_parameters)

        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1,8) /320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1,8)/320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
        output = self.transfermodule(input)
        return self.max_action * output.cpu().data.numpy().flatten()

    def initial_env_latent(self,extr,intr):
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f,cx,cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1,1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1,1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1,1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1,1)
        env_parameters = torch.cat([rv, theta, f, cx, cy], dim=1)
        self.env_latent, recon_env_parameters = self.encoder(env_parameters)
        return

    def adapt_env_latent(self, cam_obs_cur,cam_obs_goal, action_net):
        cam_obs_cur = torch.Tensor(np.array(cam_obs_cur)).to(device=device, dtype=torch.float32).view(1,8,10)/320 - 1
        cam_obs_goal = torch.Tensor(np.array(cam_obs_goal)).to(device=device, dtype=torch.float32).view(1,8,10)/320 - 1
        action_net = torch.Tensor(np.array(action_net)).to(device=device, dtype=torch.float32).view(1,6,10)/0.15
        input = torch.cat([cam_obs_cur, cam_obs_goal, action_net], dim=1)
        self.env_latent = self.adaptionmodule(input)
        return

    def select_action_with_adaptionmodule(self, cam_obs_cur,cam_obs_goal):
        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1,8) /320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1,8)/320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, self.env_latent], dim=1)
        output = self.transfermodule(input)
        return self.max_action * output.cpu().data.numpy().flatten()

    def train_transfer(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([rv, theta, f, cx, cy], dim=1)
            env_latent, recon_env_parameters = self.encoder(env_parameters)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(-1,8)/320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(-1,8)/320 - 1

            input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
            output = self.max_action * self.transfermodule(input)
            # Compute actor losse
            actor_loss = self.criterion(action_pbvs,output)
            # Optimize the actor
            self.transfer_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.transfer_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def train_adaption(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([rv, theta, f, cx, cy], dim=1)
            env_latent, recon_env_parameters = self.encoder(env_parameters)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(-1,8,10)/320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(-1,8,10)/320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32).view(-1,6,10)/0.15

            input = torch.cat([cam_obs_cur, cam_obs_goal, action_net], dim=1)
            output = self.adaptionmodule(input)
            # Compute actor losse
            actor_loss = self.criterion(env_latent,output)
            # Optimize the actor
            self.adaption_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def save(self, filename):
        torch.save(self.transfermodule.state_dict(), filename)

class AdaptionModule2(nn.Module):
    def __init__(self):
        super(AdaptionModule2, self).__init__()
        self.conv1 = torch.nn.Conv1d(8+8+6, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv1d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(40+8, 128)
        self.fc2 = torch.nn.Linear(128, 8)

    def forward(self, state, latent):
        bsz = state.shape[0]
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(bsz, -1)
        x = torch.cat([x,latent],dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TIE_Transfer4(object):
    def __init__(
            self,
            max_action,
            extr,
            intr
    ):
        self.basecontroller = BaseController().to(device=device, dtype=torch.float32)
        self.encoder = ExtIntAE(7, 8).to(device=device, dtype=torch.float32)
        self.transfermodule = SingleModule().to(device=device, dtype=torch.float32)

        self.max_action = max_action
        self.criterion = nn.MSELoss()

        self.transfer_optimizer = torch.optim.Adam([
            {'params': self.transfermodule.parameters(), 'lr': 3e-4}
        ])

        self.env_latent = 0
        self.initial_env_latent(extr, intr)
        self.adaptionmodule = AdaptionModule2().to(device=device, dtype=torch.float32)
        self.adaption_optimizer = torch.optim.Adam([
            {'params': self.adaptionmodule.parameters(), 'lr': 3e-4}
        ])

    def load_basecontroller(self, filename):
        self.basecontroller.load_state_dict(torch.load(filename + "_actor"), strict=False)

    def load_encoder(self, filename):
        self.encoder.load_state_dict(torch.load(filename), strict=False)

    def load_transfermodule(self, filename):
        self.transfermodule.load_state_dict(torch.load(filename), strict=False)

    def load_adaptionmodule(self, filename):
        self.adaptionmodule.load_state_dict(torch.load(filename), strict=False)

    def select_action_with_basecontroller(self, errstate):
        errstate = torch.Tensor(errstate).to(device=device, dtype=torch.float32).unsqueeze(0)
        return self.max_action * self.basecontroller(errstate).cpu().data.numpy().flatten()

    def select_action_with_transfermodule(self, cam_obs_cur, cam_obs_goal, extr, intr):
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([rv, theta, f, cx, cy], dim=1)
        env_latent, recon_env_parameters = self.encoder(env_parameters)

        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
        output = self.transfermodule(input)
        return self.max_action * output.cpu().data.numpy().flatten()

    def initial_env_latent(self, extr, intr):
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([rv, theta, f, cx, cy], dim=1)
        self.env_latent, recon_env_parameters = self.encoder(env_parameters)
        return

    def adapt_env_latent(self, cam_obs_cur, cam_obs_goal, action_net, last_latent):
        cam_obs_cur = torch.Tensor(np.array(cam_obs_cur)).to(device=device, dtype=torch.float32).view(1, 8,
                                                                                                      10) / 320 - 1
        cam_obs_goal = torch.Tensor(np.array(cam_obs_goal)).to(device=device, dtype=torch.float32).view(1, 8,
                                                                                                        10) / 320 - 1
        action_net = torch.Tensor(np.array(action_net)).to(device=device, dtype=torch.float32).view(1, 6, 10) / 0.15

        last_latent = torch.Tensor(np.array(last_latent)).to(device=device, dtype=torch.float32).view(1,8)
        input = torch.cat([cam_obs_cur, cam_obs_goal, action_net], dim=1)
        self.env_latent = self.adaptionmodule(input,last_latent)
        return

    def select_action_with_adaptionmodule(self, cam_obs_cur, cam_obs_goal):
        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, self.env_latent], dim=1)
        output = self.transfermodule(input)
        return self.max_action * output.cpu().data.numpy().flatten()

    def train_transfer(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([rv, theta, f, cx, cy], dim=1)
            env_latent, recon_env_parameters = self.encoder(env_parameters)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(-1, 8) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(-1, 8) / 320 - 1

            input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
            output = self.max_action * self.transfermodule(input)
            # Compute actor losse
            actor_loss = self.criterion(action_pbvs, output)
            # Optimize the actor
            self.transfer_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.transfer_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def train_adaption(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([rv, theta, f, cx, cy], dim=1)
            env_latent, recon_env_parameters = self.encoder(env_parameters)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(-1, 8, 10) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(-1, 8, 10) / 320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32).view(-1, 6, 10) / 0.15
            last_latent = batch[10].to(device=device, dtype=torch.float32).view(-1, 8)

            input = torch.cat([cam_obs_cur, cam_obs_goal, action_net], dim=1)
            output = self.adaptionmodule(input,last_latent)
            # Compute actor losse
            actor_loss = self.criterion(env_latent, output)
            # Optimize the actor
            self.adaption_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def save(self, filename):
        torch.save(self.adaptionmodule.state_dict(), filename)


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out),
        )

    def forward(self, x):
        return torch.tanh(self.mlp(x))

class VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, 32, 16)
        self.decoder = Decoder(latent_dim, 32, input_dim)
        self._enc_mu = torch.nn.Linear(16, latent_dim)
        self._enc_log_sigma = torch.nn.Linear(16, latent_dim)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(device)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * std_z  # Reparameterization trick

    def get_mean_sample(self, state, k=1):
        lis = []
        h_enc = self.encoder(state)
        if k:
            for i in range(k):
                lis.append(self._sample_latent(h_enc).cpu().detach().numpy())
            lis = np.array(lis)
            return torch.FloatTensor(np.mean(lis, axis=0))
        else:
            self._sample_latent(h_enc)  # key step for getting self.z_mean and self.z_sigma
            return self._enc_mu(h_enc)

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)

    def latent_loss(self, z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class Adaptator(nn.Module):
    def __init__(self, state_action_dim, out_dim, length=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_action_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.LeakyReLU()
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(32, 32, 5, 5),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(32, 32, 5, 1),
            nn.LeakyReLU(),
            nn.Conv1d(32, 32, 5, 1),
            nn.LeakyReLU(),
        )
        if length == 10:
            self.final = nn.Linear(32 * 2, out_dim * 2)
        elif length == 20:
            self.final = nn.Linear(32 * 12, out_dim * 2)

        self.mu_layer = nn.Linear(out_dim * 2, out_dim)
        self.std_layer = nn.Linear(out_dim * 2, out_dim)

        self.conv1 = torch.nn.Conv1d(32, 32, kernel_size=5)
        self.conv2 = torch.nn.Conv1d(32, 32, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)

    def forward(self, recent_states):
        length =recent_states.shape[1]
        res = self.mlp(recent_states).permute(0, 2, 1)
        res = self.conv1(res)
        res = F.leaky_relu(res)
        res = self.conv2(res)
        if length == 10:
            res = F.leaky_relu(res).reshape(-1, 32 * 2)
        elif length == 20:
            res = F.leaky_relu(res).reshape(-1, 32 * 12)
        res = self.final(res)
        return self._sample_latent(res)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self.mu_layer(h_enc)
        log_sigma = self.std_layer(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(device)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * std_z  # Reparameterization trick

    def get_mean_sample(self, recent_states, k=10):
        lis = []
        res = self.mlp(recent_states).permute(0, 2, 1)
        res = self.cnn(res).reshape(-1, 32 * 3)
        h_enc = self.final(res)
        if k:
            if k == 1:
                return self._sample_latent(h_enc)
            else:
                for i in range(k):
                    lis.append(self._sample_latent(h_enc).cpu().detach().numpy())
                lis = np.array(lis)
                return torch.FloatTensor(np.mean(lis, axis=0))
        else:
            return self.mu_layer(h_enc)

    def latent_loss(self, z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class TIE_Transfer_VAE(object):
    def __init__(
            self,
            max_action,
            extr,
            intr
    ):
        self.basecontroller = BaseController().to(device=device, dtype=torch.float32)
        self.encoder = VAE(10, 8).to(device=device, dtype=torch.float32)
        self.transfermodule = SingleModule().to(device=device, dtype=torch.float32)

        self.max_action = max_action
        self.criterion = nn.MSELoss()
        self.adaption_criterion = nn.MSELoss(reduction='sum')
        self.transfer_optimizer = torch.optim.Adam([
            {'params': self.transfermodule.parameters(), 'lr': 3e-4}
        ])

        self.env_latent = 0
        self.initial_env_latent(extr, intr)
        self.adaptionmodule = Adaptator(8+8+6,8).to(device=device, dtype=torch.float32)
        self.adaption_optimizer = torch.optim.Adam([
            {'params': self.adaptionmodule.parameters(), 'lr': 3e-4}
        ])

    def load_basecontroller(self, filename):
        self.basecontroller.load_state_dict(torch.load(filename + "_actor"), strict=False)

    def load_encoder(self, filename):
        self.encoder.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    def load_transfermodule(self, filename):
        self.transfermodule.load_state_dict(torch.load(filename), strict=False)

    def load_adaptionmodule(self, filename):
        self.adaptionmodule.load_state_dict(torch.load(filename), strict=False)

    def select_action_with_basecontroller(self, errstate):
        errstate = torch.Tensor(errstate).to(device=device, dtype=torch.float32).unsqueeze(0)
        return self.max_action * self.basecontroller(errstate).cpu().data.numpy().flatten()

    def select_action_with_transfermodule(self, cam_obs_cur, cam_obs_goal, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([translation, rv, theta, f, cx, cy], dim=1)
        env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
        output = self.transfermodule(input)
        return self.max_action * output.cpu().data.numpy().flatten()

    def initial_env_latent(self, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([translation, rv, theta, f, cx, cy], dim=1)
        self.env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)
        return

    def adapt_env_latent(self, cam_obs_cur, cam_obs_goal, action_net, last_latent, length=10):
        cam_obs_cur = torch.Tensor(np.array(cam_obs_cur)).to(device=device, dtype=torch.float32).view(1, 8,
                                                                                                      length) / 320 - 1
        cam_obs_goal = torch.Tensor(np.array(cam_obs_goal)).to(device=device, dtype=torch.float32).view(1, 8,
                                                                                                        length) / 320 - 1
        action_net = torch.Tensor(np.array(action_net)).to(device=device, dtype=torch.float32).view(1, 6, length) / 0.15

        last_latent = torch.Tensor(np.array(last_latent)).to(device=device, dtype=torch.float32).view(1,8)
        input = torch.cat([cam_obs_cur, cam_obs_goal, action_net], dim=1).permute(0,2,1)
        self.env_latent = self.adaptionmodule(input).to(device=device, dtype=torch.float32)
        return

    def select_action_with_adaptionmodule(self, cam_obs_cur, cam_obs_goal):
        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, self.env_latent], dim=1)
        output = self.transfermodule(input)
        return self.max_action * output.cpu().data.numpy().flatten()

    def train_transfer(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[0].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([translation, rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8) / 320 - 1

            input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
            output = self.max_action * self.transfermodule(input)
            # Compute actor losse
            actor_loss = self.criterion(action_pbvs, output)
            # Optimize the actor
            self.transfer_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.transfer_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def save_transfer(self, filename):
        torch.save(self.transfermodule.state_dict(), filename)

    def train_adaption(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[1].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([translation, rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32).view(bsz, 6, -1) / 0.15
            last_latent = batch[10].to(device=device, dtype=torch.float32).view(bsz, 8)

            input = torch.cat([cam_obs_cur, cam_obs_goal, action_net], dim=1).permute(0,2,1)
            output = self.adaptionmodule(input)
            # Compute actor loss
            actor_loss = self.criterion(env_latent, output)
            # ll = self.adaptionmodule.latent_loss(self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma)
            # pre_mu_std = torch.cat((self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma), axis=1)
            # true_mu_std = torch.cat((self.encoder.z_mean, self.encoder.z_sigma), axis=1)
            # ml = self.adaption_criterion(pre_mu_std, true_mu_std)
            # actor_loss = ll + ml
            # Optimize the actor
            self.adaption_optimizer.zero_grad()

            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def save_adaption(self, filename):
        torch.save(self.adaptionmodule.state_dict(), filename)

class TIE_Transfer_VAE7(object):
    def __init__(
            self,
            max_action,
            extr,
            intr,
            length=10
    ):
        self.basecontroller = BaseController().to(device=device, dtype=torch.float32)
        self.encoder = VAE(7, 8).to(device=device, dtype=torch.float32)
        self.transfermodule = SingleModule().to(device=device, dtype=torch.float32)

        self.max_action = max_action
        self.criterion = nn.MSELoss()
        self.adaption_criterion = nn.MSELoss(reduction='sum')
        self.transfer_optimizer = torch.optim.Adam([
            {'params': self.transfermodule.parameters(), 'lr': 3e-4}
        ])

        self.env_latent = 0
        self.initial_env_latent(extr, intr)
        # self.adaptionmodule = Adaptator(8+8+6,8,length=length).to(device=device, dtype=torch.float32)
        self.adaptionmodule = Adaptator(8+6,8,length=length).to(device=device, dtype=torch.float32)
        self.adaption_optimizer = torch.optim.Adam([
            {'params': self.adaptionmodule.parameters(), 'lr': 3e-4}
        ])

    def load_basecontroller(self, filename):
        self.basecontroller.load_state_dict(torch.load(filename + "_actor"), strict=False)

    def load_encoder(self, filename):
        self.encoder.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    def load_transfermodule(self, filename):
        self.transfermodule.load_state_dict(torch.load(filename), strict=False)

    def load_adaptionmodule(self, filename):
        self.adaptionmodule.load_state_dict(torch.load(filename), strict=False)

    def select_action_with_basecontroller(self, errstate):
        errstate = torch.Tensor(errstate).to(device=device, dtype=torch.float32).unsqueeze(0)
        return self.max_action * self.basecontroller(errstate).cpu().data.numpy().flatten()

    def select_action_with_transfermodule(self, cam_obs_cur, cam_obs_goal, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
        env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
        output = self.transfermodule(input)
        return self.max_action * output.cpu().data.numpy().flatten()

    def initial_env_latent(self, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
        self.env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)
        return

    def adapt_env_latent(self, cam_obs_cur, cam_obs_goal, action_net, last_latent, length=10):
        cam_obs_cur = torch.Tensor(np.array(cam_obs_cur)).to(device=device, dtype=torch.float32).view(1, 8,
                                                                                                      length) / 320 - 1
        cam_obs_goal = torch.Tensor(np.array(cam_obs_goal)).to(device=device, dtype=torch.float32).view(1, 8,
                                                                                                        length) / 320 - 1
        action_net = torch.Tensor(np.array(action_net)).to(device=device, dtype=torch.float32).view(1, 6, length) / 0.15

        last_latent = torch.Tensor(np.array(last_latent)).to(device=device, dtype=torch.float32).view(1,8)
        # input = torch.cat([cam_obs_cur, cam_obs_goal, action_net], dim=1).permute(0,2,1)
        input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0, 2, 1)
        self.env_latent = self.adaptionmodule(input).to(device=device, dtype=torch.float32)
        return

    def select_action_with_adaptionmodule(self, cam_obs_cur, cam_obs_goal):
        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, self.env_latent], dim=1)
        output = self.transfermodule(input)
        return self.max_action * output.cpu().data.numpy().flatten()

    def train_transfer(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[0].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8) / 320 - 1

            input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
            output = self.max_action * self.transfermodule(input)
            # Compute actor losse
            actor_loss = self.criterion(action_pbvs, output)
            # Optimize the actor
            self.transfer_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.transfer_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def save_transfer(self, filename):
        torch.save(self.transfermodule.state_dict(), filename)

    def train_adaption(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[1].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32).view(bsz, 6, -1) / 0.15
            last_latent = batch[10].to(device=device, dtype=torch.float32).view(bsz, 8)

            # input = torch.cat([cam_obs_cur, cam_obs_goal, action_net], dim=1).permute(0,2,1)
            input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0,2,1)
            output = self.adaptionmodule(input)
            # Compute actor loss
            actor_loss = self.criterion(env_latent, output)
            # ll = self.adaptionmodule.latent_loss(self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma)
            # pre_mu_std = torch.cat((self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma), axis=1)
            # true_mu_std = torch.cat((self.encoder.z_mean, self.encoder.z_sigma), axis=1)
            # ml = self.adaption_criterion(pre_mu_std, true_mu_std)
            # actor_loss = ml
            # actor_loss = ll + ml
            # Optimize the actor
            self.adaption_optimizer.zero_grad()

            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def save_adaption(self, filename):
        torch.save(self.adaptionmodule.state_dict(), filename)

class SingleModule_caz(nn.Module):
    def __init__(self):
        super(SingleModule_caz, self).__init__()
        self.l1 = nn.Linear(8+8+8, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 6)
        self.l5 = nn.Linear(128, 1)

    def forward(self, state):
        bsz = state.shape[0]
        input = state.view(bsz,-1,)
        a = F.relu(self.l1(input))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return torch.tanh(self.l4(a)),-F.relu(self.l5(a))

def scaled_cosine_similarity_loss(pred, target):
    similarity = F.cosine_similarity(pred, target, dim=-1)
    loss = 1 - similarity
    return loss

def reverse_alpha_p_elu(y: torch.Tensor, alpha=1.0, eps=1e-7):
    mask = y < alpha
    if mask.ndim == 0:  # is scalar
        x = torch.log(y/alpha + eps) if mask else (y - alpha)
    else:
        x = y - alpha
        x[mask] = torch.log(y[mask] / alpha + eps)
    return x

def alpha_p_elu(x, alpha=1.0):
    return alpha + F.elu(x, alpha=alpha)

class Adaptator_caz(nn.Module):
    def __init__(self, state_action_dim, out_dim, length=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_action_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.LeakyReLU()
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(32, 32, 5, 5),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(32, 32, 5, 1),
            nn.LeakyReLU(),
            nn.Conv1d(32, 32, 5, 1),
            nn.LeakyReLU(),
        )
        if length == 10:
            self.final = nn.Linear(32 * 2, out_dim * 2)
        elif length == 20:
            self.final = nn.Linear(32 * 12, out_dim * 2)

        self.mu_layer = nn.Linear(out_dim * 2, out_dim)
        self.std_layer = nn.Linear(out_dim * 2, out_dim)

        self.conv1 = torch.nn.Conv1d(32, 32, kernel_size=5)
        self.conv2 = torch.nn.Conv1d(32, 32, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)

    def forward(self, recent_states):
        length =recent_states.shape[1]
        res = self.mlp(recent_states).permute(0, 2, 1)
        res = self.conv1(res)
        res = F.leaky_relu(res)
        res = self.conv2(res)
        if length == 10:
            res = F.leaky_relu(res).reshape(-1, 32 * 2)
        elif length == 20:
            res = F.leaky_relu(res).reshape(-1, 32 * 12)
        res = self.final(res)
        return self._sample_latent(res)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self.mu_layer(h_enc)
        log_sigma = self.std_layer(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(device)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * std_z  # Reparameterization trick

    def get_mean_sample(self, recent_states, k=10):
        lis = []
        res = self.mlp(recent_states).permute(0, 2, 1)
        res = self.cnn(res).reshape(-1, 32 * 3)
        h_enc = self.final(res)
        if k:
            if k == 1:
                return self._sample_latent(h_enc)
            else:
                for i in range(k):
                    lis.append(self._sample_latent(h_enc).cpu().detach().numpy())
                lis = np.array(lis)
                return torch.FloatTensor(np.mean(lis, axis=0))
        else:
            return self.mu_layer(h_enc)

    def latent_loss(self, z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class Adaptator_caz_conv2d(nn.Module):
    def __init__(self, state_action_dim, out_dim, length=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_action_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.LeakyReLU()
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(32, 32, 5, 5),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(32, 32, 5, 1),
            nn.LeakyReLU(),
            nn.Conv1d(32, 32, 5, 1),
            nn.LeakyReLU(),
        )
        if length == 10:
            self.final = nn.Linear(64 * 13 * 2, out_dim * 2)
        elif length == 20:
            self.final = nn.Linear(32 * 12, out_dim * 2)

        self.mu_layer = nn.Linear(out_dim * 2, out_dim)
        self.std_layer = nn.Linear(out_dim * 2, out_dim)

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.pooling = torch.nn.AvgPool2d(2)

    def forward(self, recent_states):
        bsz = recent_states.shape[0]
        length =recent_states.shape[1]
        res = self.mlp(recent_states).permute(0, 2, 1).view(bsz,1,32,length)
        res = self.conv1(res)
        res = F.leaky_relu(res)
        res = self.pooling(res)
        res = self.conv2(res)
        if length == 10:
            res = F.leaky_relu(res).reshape(bsz, 64 * 13 * 2)
        elif length == 20:
            res = F.leaky_relu(res).reshape(bsz, 32 * 12)
        res = self.final(res)
        return self._sample_latent(res)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self.mu_layer(h_enc)
        log_sigma = self.std_layer(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(device)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * std_z  # Reparameterization trick

    def get_mean_sample(self, recent_states, k=10):
        lis = []
        res = self.mlp(recent_states).permute(0, 2, 1)
        res = self.cnn(res).reshape(-1, 32 * 3)
        h_enc = self.final(res)
        if k:
            if k == 1:
                return self._sample_latent(h_enc)
            else:
                for i in range(k):
                    lis.append(self._sample_latent(h_enc).cpu().detach().numpy())
                lis = np.array(lis)
                return torch.FloatTensor(np.mean(lis, axis=0))
        else:
            return self.mu_layer(h_enc)

    def latent_loss(self, z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class TIE_Transfer_VAE7_caz(object):
    def __init__(
            self,
            max_action,
            extr,
            intr,
            length=10
    ):
        self.basecontroller = BaseController().to(device=device, dtype=torch.float32)
        self.encoder = VAE(7, 8).to(device=device, dtype=torch.float32)
        self.transfermodule = SingleModule_caz().to(device=device, dtype=torch.float32)

        self.max_action = max_action
        self.criterion = nn.MSELoss()
        self.adaption_criterion = nn.MSELoss(reduction='sum')
        self.transfer_optimizer = torch.optim.Adam([
            {'params': self.transfermodule.parameters(), 'lr': 3e-4}
        ])

        self.env_latent = 0
        self.initial_env_latent(extr, intr)
        # self.adaptionmodule = Adaptator(8+8+6,8,length=length).to(device=device, dtype=torch.float32)
        self.adaptionmodule = Adaptator_caz_conv2d(8+8+7,8,length=length).to(device=device, dtype=torch.float32)
        self.adaption_optimizer = torch.optim.Adam([
            {'params': self.adaptionmodule.parameters(), 'lr': 3e-4}
        ])

    def load_basecontroller(self, filename):
        self.basecontroller.load_state_dict(torch.load(filename + "_actor"), strict=False)

    def load_encoder(self, filename):
        self.encoder.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    def load_transfermodule(self, filename):
        self.transfermodule.load_state_dict(torch.load(filename,map_location=lambda storage, loc: storage), strict=False)

    def load_adaptionmodule(self, filename):
        self.adaptionmodule.load_state_dict(torch.load(filename), strict=False)

    def select_action_with_basecontroller(self, errstate):
        errstate = torch.Tensor(errstate).to(device=device, dtype=torch.float32).unsqueeze(0)
        return self.max_action * self.basecontroller(errstate).cpu().data.numpy().flatten()

    def select_action_with_transfermodule(self, cam_obs_cur, cam_obs_goal, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
        env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
        vel_vec, vel_norm = self.transfermodule(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten()

    def initial_env_latent(self, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
        self.env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)
        return

    def adapt_env_latent(self, cam_obs_cur, cam_obs_goal, action_net, last_latent, length=10):
        cam_obs_cur = torch.Tensor(np.array(cam_obs_cur)).to(device=device, dtype=torch.float32).view(1, 8,
                                                                                                      length) / 320 - 1
        cam_obs_goal = torch.Tensor(np.array(cam_obs_goal)).to(device=device, dtype=torch.float32).view(1, 8,
                                                                                                        length) / 320 - 1
        action_net = torch.Tensor(np.array(action_net)).to(device=device, dtype=torch.float32)

        last_latent = torch.Tensor(np.array(last_latent)).to(device=device, dtype=torch.float32).view(1,8)
        # input = torch.cat([cam_obs_cur, cam_obs_goal, action_net], dim=1).permute(0,2,1)
        # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0, 2, 1)
        action_norm = torch.norm(action_net, dim=1, keepdim=True)
        action_vec = action_net / (action_norm + 1e-7)

        input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(1, 6,length), action_norm.view(1, 1,length)], dim=1).permute(0, 2, 1)
        self.env_latent = self.adaptionmodule(input).to(device=device, dtype=torch.float32)
        return

    def select_action_with_adaptionmodule(self, cam_obs_cur, cam_obs_goal):
        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, self.env_latent], dim=1)
        vel_vec, vel_norm = self.transfermodule(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten(), vel_vec.cpu().data.numpy().flatten(), vel_norm.cpu().data.numpy().flatten()

    def train_transfer(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        total_dir_loss = 0
        total_norm_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[0].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            gt = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8) / 320 - 1

            input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
            vel_vec, vel_norm = self.transfermodule(input)
            # Compute actor losse
            dir_loss = scaled_cosine_similarity_loss(vel_vec, gt).mean()
            gt_norm = torch.norm(gt, dim=-1, keepdim=True)
            norm_loss = F.mse_loss(
                vel_norm,
                reverse_alpha_p_elu(gt_norm, eps=1e-5),
                reduction="mean"
            ) * 0.5
            actor_loss = dir_loss + norm_loss
            # Optimize the actor
            self.transfer_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.transfer_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            total_dir_loss += dir_loss
            total_norm_loss += norm_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
            tb.add_scalar('average_dir_loss', total_dir_loss / count, episode_timesteps)
            tb.add_scalar('average_norm_loss', total_norm_loss / count, episode_timesteps)
        return

    def save_transfer(self, filename):
        torch.save(self.transfermodule.state_dict(), filename)

    def train_adaption(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[1].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32)
            last_latent = batch[10].to(device=device, dtype=torch.float32).view(bsz, 8)

            action_norm = torch.norm(action_net, dim=-1, keepdim=True)
            action_vec = action_net / (action_norm + 1e-7)

            input = torch.cat(
                [cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1), action_norm.view(bsz, 1, -1)],
                dim=1).permute(0, 2, 1)


            # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0,2,1)
            output = self.adaptionmodule(input)
            # Compute actor loss
            actor_loss = self.criterion(env_latent, output)
            # ll = self.adaptionmodule.latent_loss(self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma)
            # pre_mu_std = torch.cat((self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma), axis=1)
            # true_mu_std = torch.cat((self.encoder.z_mean, self.encoder.z_sigma), axis=1)
            # ml = self.adaption_criterion(pre_mu_std, true_mu_std)
            # actor_loss = ml
            # actor_loss = ll + ml
            # Optimize the actor
            self.adaption_optimizer.zero_grad()

            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def save_adaption(self, filename):
        torch.save(self.adaptionmodule.state_dict(), filename)

class TIE_Transfer_VAE7_caz_tcp(object):
    def __init__(
            self,
            max_action,
            extr,
            intr,
            length=10
    ):
        self.basecontroller = BaseController().to(device=device, dtype=torch.float32)
        self.encoder = VAE(7, 8).to(device=device, dtype=torch.float32)
        self.transfermodule = SingleModule_caz().to(device=device, dtype=torch.float32)

        self.max_action = max_action
        self.criterion = nn.MSELoss()
        self.adaption_criterion = nn.MSELoss(reduction='sum')
        self.transfer_optimizer = torch.optim.Adam([
            {'params': self.transfermodule.parameters(), 'lr': 3e-4}
        ])

        self.env_latent = 0
        self.initial_env_latent(extr, intr)
        # self.adaptionmodule = Adaptator(8+8+6,8,length=length).to(device=device, dtype=torch.float32)
        self.adaptionmodule = Adaptator_caz_conv2d(8+8+7+7,8,length=length).to(device=device, dtype=torch.float32)
        self.adaption_optimizer = torch.optim.Adam([
            {'params': self.adaptionmodule.parameters(), 'lr': 3e-4}
        ])

    def load_basecontroller(self, filename):
        self.basecontroller.load_state_dict(torch.load(filename + "_actor"), strict=False)

    def load_encoder(self, filename):
        self.encoder.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    def load_transfermodule(self, filename):
        self.transfermodule.load_state_dict(torch.load(filename), strict=False)

    def load_adaptionmodule(self, filename):
        self.adaptionmodule.load_state_dict(torch.load(filename), strict=False)

    def select_action_with_basecontroller(self, errstate):
        errstate = torch.Tensor(errstate).to(device=device, dtype=torch.float32).unsqueeze(0)
        return self.max_action * self.basecontroller(errstate).cpu().data.numpy().flatten()

    def select_action_with_transfermodule(self, cam_obs_cur, cam_obs_goal, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
        env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
        vel_vec, vel_norm = self.transfermodule(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten()

    def initial_env_latent(self, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
        self.env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)
        return

    def adapt_env_latent(self, cam_obs_cur, cam_obs_goal, action_net, tcp_pose, last_latent,  length=10):
        cam_obs_cur = torch.Tensor(np.array(cam_obs_cur)).to(device=device, dtype=torch.float32).view(1, 8,
                                                                                                      length) / 320 - 1
        cam_obs_goal = torch.Tensor(np.array(cam_obs_goal)).to(device=device, dtype=torch.float32).view(1, 8,
                                                                                                        length) / 320 - 1
        action_net = torch.Tensor(np.array(action_net)).to(device=device, dtype=torch.float32)

        tcp_rv = []
        tcp_theta = []
        for i in range(tcp_pose.shape[0]):
            tcp_rv.append(R.from_matrix(tcp_pose[i, :3, :3]).as_rotvec())
            tcp_theta.append(np.linalg.norm(tcp_rv[i]))
            tcp_rv[i] = tcp_rv[i]/(tcp_theta[i] + 1e-7)
        tcp_translation = torch.Tensor(tcp_pose[:, :3, 3]).to(device=device, dtype=torch.float32).view(1, 3, length)
        tcp_rv = torch.Tensor(np.array(tcp_rv)).to(device=device, dtype=torch.float32).view(1, 3, length)
        tcp_theta = torch.Tensor(np.array(tcp_theta)).to(device=device, dtype=torch.float32).view(1, 1, length)

        last_latent = torch.Tensor(np.array(last_latent)).to(device=device, dtype=torch.float32).view(1,8)
        # input = torch.cat([cam_obs_cur, cam_obs_goal, action_net], dim=1).permute(0,2,1)
        # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0, 2, 1)
        action_norm = torch.norm(action_net, dim=1, keepdim=True)
        action_vec = action_net / (action_norm + 1e-7)

        input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(1, 6,length),
                           action_norm.view(1, 1,length),tcp_translation,tcp_rv,tcp_theta], dim=1).permute(0, 2, 1)
        self.env_latent = self.adaptionmodule(input).to(device=device, dtype=torch.float32)
        return

    def select_action_with_adaptionmodule(self, cam_obs_cur, cam_obs_goal):
        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, self.env_latent], dim=1)
        vel_vec, vel_norm = self.transfermodule(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten(), vel_vec.cpu().data.numpy().flatten(), vel_norm.cpu().data.numpy().flatten()

    def train_transfer(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        total_dir_loss = 0
        total_norm_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[0].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            gt = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8) / 320 - 1

            input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
            vel_vec, vel_norm = self.transfermodule(input)
            # Compute actor losse
            dir_loss = scaled_cosine_similarity_loss(vel_vec, gt).mean()
            gt_norm = torch.norm(gt, dim=-1, keepdim=True)
            norm_loss = F.mse_loss(
                vel_norm,
                reverse_alpha_p_elu(gt_norm, eps=1e-5),
                reduction="mean"
            ) * 0.5
            actor_loss = dir_loss + norm_loss
            # Optimize the actor
            self.transfer_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.transfer_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            total_dir_loss += dir_loss
            total_norm_loss += norm_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
            tb.add_scalar('average_dir_loss', total_dir_loss / count, episode_timesteps)
            tb.add_scalar('average_norm_loss', total_norm_loss / count, episode_timesteps)
        return

    def save_transfer(self, filename):
        torch.save(self.transfermodule.state_dict(), filename)

    def train_adaption(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[1].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32)
            last_latent = batch[10].to(device=device, dtype=torch.float32).view(bsz, 8)

            tcp_rv = batch[11].to(device=device, dtype=torch.float32).view(bsz, 3, -1)
            tcp_theta = batch[12].to(device=device, dtype=torch.float32).unsqueeze(1)
            tcp_translation = batch[13].to(device=device, dtype=torch.float32).view(bsz, 3, -1)

            action_norm = torch.norm(action_net, dim=-1, keepdim=True)
            action_vec = action_net / (action_norm + 1e-7)

            # input = torch.cat(
            #     [cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1), action_norm.view(bsz, 1, -1)],
            #     dim=1).permute(0, 2, 1)
            input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1),
                               action_norm.view(bsz, 1, -1), tcp_translation, tcp_rv, tcp_theta], dim=1).permute(0, 2,
                                                                                                                   1)



            # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0,2,1)
            output = self.adaptionmodule(input)
            # Compute actor loss
            actor_loss = self.criterion(env_latent, output)
            # ll = self.adaptionmodule.latent_loss(self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma)
            # pre_mu_std = torch.cat((self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma), axis=1)
            # true_mu_std = torch.cat((self.encoder.z_mean, self.encoder.z_sigma), axis=1)
            # ml = self.adaption_criterion(pre_mu_std, true_mu_std)
            # actor_loss = ml
            # actor_loss = ll + ml
            # Optimize the actor
            self.adaption_optimizer.zero_grad()

            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def train_adaption_velloss(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[1].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            gt = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32)
            last_latent = batch[10].to(device=device, dtype=torch.float32).view(bsz, 8)

            tcp_rv = batch[11].to(device=device, dtype=torch.float32).view(bsz, 3, -1)
            tcp_theta = batch[12].to(device=device, dtype=torch.float32).unsqueeze(1)
            tcp_translation = batch[13].to(device=device, dtype=torch.float32).view(bsz, 3, -1)

            action_norm = torch.norm(action_net, dim=-1, keepdim=True)
            action_vec = action_net / (action_norm + 1e-7)

            # input = torch.cat(
            #     [cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1), action_norm.view(bsz, 1, -1)],
            #     dim=1).permute(0, 2, 1)
            input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1),
                               action_norm.view(bsz, 1, -1), tcp_translation, tcp_rv, tcp_theta], dim=1).permute(0, 2,
                                                                                                                   1)



            # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0,2,1)
            output = self.adaptionmodule(input)
            # Compute actor loss
            actor_loss = self.criterion(env_latent, output)
            # ll = self.adaptionmodule.latent_loss(self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma)
            # pre_mu_std = torch.cat((self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma), axis=1)
            # true_mu_std = torch.cat((self.encoder.z_mean, self.encoder.z_sigma), axis=1)
            # ml = self.adaption_criterion(pre_mu_std, true_mu_std)
            # actor_loss = ml
            # actor_loss = ll + ml
            # Optimize the actor
            self.adaption_optimizer.zero_grad()

            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def save_adaption(self, filename):
        torch.save(self.adaptionmodule.state_dict(), filename)

class TIE_Transfer_VAE7_caz_tcp_multihead(object):
    def __init__(
            self,
            max_action,
            extr,
            intr,
            length=10
    ):
        self.basecontroller = BaseController().to(device=device, dtype=torch.float32)
        self.encoder = VAE(7, 8).to(device=device, dtype=torch.float32)
        self.transfermodule = SingleModule_caz().to(device=device, dtype=torch.float32)

        self.max_action = max_action
        self.criterion = nn.MSELoss()
        self.adaption_criterion = nn.MSELoss(reduction='sum')
        self.transfer_optimizer = torch.optim.Adam([
            {'params': self.transfermodule.parameters(), 'lr': 3e-4}
        ])

        self.env_latent = 0
        self.initial_env_latent(extr, intr)
        # self.adaptionmodule = Adaptator(8+8+6,8,length=length).to(device=device, dtype=torch.float32)
        self.adaptionmodule = Adaptator_caz_conv2d_multihead([8,8,3,3,1],8,length=length).to(device=device, dtype=torch.float32)
        self.adaption_optimizer = torch.optim.Adam([
            {'params': self.adaptionmodule.parameters(), 'lr': 3e-4}
        ])

    def load_basecontroller(self, filename):
        self.basecontroller.load_state_dict(torch.load(filename + "_actor"), strict=False)

    def load_encoder(self, filename):
        self.encoder.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    def load_transfermodule(self, filename):
        self.transfermodule.load_state_dict(torch.load(filename), strict=False)

    def load_adaptionmodule(self, filename):
        self.adaptionmodule.load_state_dict(torch.load(filename), strict=False)

    def select_action_with_basecontroller(self, errstate):
        errstate = torch.Tensor(errstate).to(device=device, dtype=torch.float32).unsqueeze(0)
        return self.max_action * self.basecontroller(errstate).cpu().data.numpy().flatten()

    def select_action_with_transfermodule(self, cam_obs_cur, cam_obs_goal, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
        env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
        vel_vec, vel_norm = self.transfermodule(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten()

    def initial_env_latent(self, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
        self.env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)
        return

    def adapt_env_latent(self, cam_obs_cur, cam_obs_goal, action_net, tcp_pose, last_latent,  length=10):
        cam_obs_cur = torch.Tensor(np.array(cam_obs_cur)).to(device=device, dtype=torch.float32).view(1, 8,
                                                                                                      length) / 320 - 1
        cam_obs_goal = torch.Tensor(np.array(cam_obs_goal)).to(device=device, dtype=torch.float32).view(1, 8,
                                                                                                        length) / 320 - 1
        action_net = torch.Tensor(np.array(action_net)).to(device=device, dtype=torch.float32)

        tcp_rv = []
        tcp_theta = []
        for i in range(tcp_pose.shape[0]):
            tcp_rv.append(R.from_matrix(tcp_pose[i, :3, :3]).as_rotvec())
            tcp_theta.append(np.linalg.norm(tcp_rv[i]))
            tcp_rv[i] = tcp_rv[i]/(tcp_theta[i] + 1e-7)
        tcp_translation = torch.Tensor(tcp_pose[:, :3, 3]).to(device=device, dtype=torch.float32).view(1, 3, length)
        tcp_rv = torch.Tensor(np.array(tcp_rv)).to(device=device, dtype=torch.float32).view(1, 3, length)
        tcp_theta = torch.Tensor(np.array(tcp_theta)).to(device=device, dtype=torch.float32).view(1, 1, length)

        last_latent = torch.Tensor(np.array(last_latent)).to(device=device, dtype=torch.float32).view(1,8)
        # input = torch.cat([cam_obs_cur, cam_obs_goal, action_net], dim=1).permute(0,2,1)
        # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0, 2, 1)
        action_norm = torch.norm(action_net, dim=1, keepdim=True)
        action_vec = action_net / (action_norm + 1e-7)

        # input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(1, 6,length),
        #                    action_norm.view(1, 1,length),tcp_translation,tcp_rv,tcp_theta], dim=1).permute(0, 2, 1)

        self.env_latent = self.adaptionmodule(cam_obs_cur.permute(0, 2, 1), cam_obs_goal.permute(0, 2, 1),
                                     tcp_translation.permute(0, 2, 1), tcp_rv.permute(0, 2, 1),
                                     tcp_theta.permute(0, 2, 1)).to(device=device, dtype=torch.float32)
        return

    def select_action_with_adaptionmodule(self, cam_obs_cur, cam_obs_goal):
        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, self.env_latent], dim=1)
        vel_vec, vel_norm = self.transfermodule(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten(), vel_vec.cpu().data.numpy().flatten(), vel_norm.cpu().data.numpy().flatten()

    def train_transfer(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        total_dir_loss = 0
        total_norm_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[0].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            gt = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8) / 320 - 1

            input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
            vel_vec, vel_norm = self.transfermodule(input)
            # Compute actor losse
            dir_loss = scaled_cosine_similarity_loss(vel_vec, gt).mean()
            gt_norm = torch.norm(gt, dim=-1, keepdim=True)
            norm_loss = F.mse_loss(
                vel_norm,
                reverse_alpha_p_elu(gt_norm, eps=1e-5),
                reduction="mean"
            ) * 0.5
            actor_loss = dir_loss + norm_loss
            # Optimize the actor
            self.transfer_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.transfer_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            total_dir_loss += dir_loss
            total_norm_loss += norm_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
            tb.add_scalar('average_dir_loss', total_dir_loss / count, episode_timesteps)
            tb.add_scalar('average_norm_loss', total_norm_loss / count, episode_timesteps)
        return

    def save_transfer(self, filename):
        torch.save(self.transfermodule.state_dict(), filename)

    def train_adaption(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[1].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32)
            last_latent = batch[10].to(device=device, dtype=torch.float32).view(bsz, 8)

            tcp_rv = batch[11].to(device=device, dtype=torch.float32).view(bsz, 3, -1)
            tcp_theta = batch[12].to(device=device, dtype=torch.float32).unsqueeze(1)
            tcp_translation = batch[13].to(device=device, dtype=torch.float32).view(bsz, 3, -1)

            action_norm = torch.norm(action_net, dim=-1, keepdim=True)
            action_vec = action_net / (action_norm + 1e-7)

            # input = torch.cat(
            #     [cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1), action_norm.view(bsz, 1, -1)],
            #     dim=1).permute(0, 2, 1)

            # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0,2,1)
            output = self.adaptionmodule(cam_obs_cur.permute(0, 2,1), cam_obs_goal.permute(0, 2,1), tcp_translation.permute(0, 2,1), tcp_rv.permute(0, 2,1), tcp_theta.permute(0, 2,1))
            # Compute actor loss
            actor_loss = self.criterion(env_latent, output)
            # ll = self.adaptionmodule.latent_loss(self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma)
            # pre_mu_std = torch.cat((self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma), axis=1)
            # true_mu_std = torch.cat((self.encoder.z_mean, self.encoder.z_sigma), axis=1)
            # ml = self.adaption_criterion(pre_mu_std, true_mu_std)
            # actor_loss = ml
            # actor_loss = ll + ml
            # Optimize the actor
            self.adaption_optimizer.zero_grad()

            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def save_adaption(self, filename):
        torch.save(self.adaptionmodule.state_dict(), filename)

class Adaptator_caz_conv2d_multihead(nn.Module):
    def __init__(self, sad, out_dim, length=10):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(sad[0], 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 8),
            nn.LeakyReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(sad[1], 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 8),
            nn.LeakyReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(sad[2], 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6),
            nn.LeakyReLU()
        )
        self.mlp4 = nn.Sequential(
            nn.Linear(sad[3], 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6),
            nn.LeakyReLU()
        )
        self.mlp5 = nn.Sequential(
            nn.Linear(sad[4], 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4),
            nn.LeakyReLU()
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(32, 32, 5, 5),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(32, 32, 5, 1),
            nn.LeakyReLU(),
            nn.Conv1d(32, 32, 5, 1),
            nn.LeakyReLU(),
        )
        if length == 10:
            self.final = nn.Linear(64 * 13 * 2, out_dim * 2)
        elif length == 20:
            self.final = nn.Linear(32 * 12, out_dim * 2)

        self.mu_layer = nn.Linear(out_dim * 2, out_dim)
        self.std_layer = nn.Linear(out_dim * 2, out_dim)

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.pooling = torch.nn.AvgPool2d(2)

    def forward(self,cam_obs_cur, cam_obs_goal, tcp_translation, tcp_rv, tcp_theta):
        bsz = cam_obs_cur.shape[0]
        length =cam_obs_cur.shape[1]

        res1 = self.mlp1(cam_obs_cur).permute(0, 2, 1)#.view(bsz,1,32,length)
        res2 = self.mlp2(cam_obs_goal).permute(0, 2, 1)#.view(bsz,1,32,length)
        res3 = self.mlp3(tcp_translation).permute(0, 2, 1)#.view(bsz,1,32,length)
        res4 = self.mlp4(tcp_rv).permute(0, 2, 1)#.view(bsz,1,32,length)
        res5 = self.mlp5(tcp_theta).permute(0, 2, 1)#.view(bsz,1,32,length)

        res = torch.cat([res1,res2,res3,res4,res5],dim=1).view(bsz,1,32,length)
        res = self.conv1(res)
        res = F.leaky_relu(res)
        res = self.pooling(res)
        res = self.conv2(res)
        if length == 10:
            res = F.leaky_relu(res).reshape(bsz, 64 * 13 * 2)
        elif length == 20:
            res = F.leaky_relu(res).reshape(bsz, 32 * 12)
        res = self.final(res)
        return self._sample_latent(res)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self.mu_layer(h_enc)
        log_sigma = self.std_layer(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(device)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * std_z  # Reparameterization trick

    def get_mean_sample(self, recent_states, k=10):
        lis = []
        res = self.mlp(recent_states).permute(0, 2, 1)
        res = self.cnn(res).reshape(-1, 32 * 3)
        h_enc = self.final(res)
        if k:
            if k == 1:
                return self._sample_latent(h_enc)
            else:
                for i in range(k):
                    lis.append(self._sample_latent(h_enc).cpu().detach().numpy())
                lis = np.array(lis)
                return torch.FloatTensor(np.mean(lis, axis=0))
        else:
            return self.mu_layer(h_enc)

    def latent_loss(self, z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class Adaptator_caz_conv2d_multihead_prompt(nn.Module):
    def __init__(self, sad, out_dim, length=10):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(sad[0], 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 8),
            nn.LeakyReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(sad[1], 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 8),
            nn.LeakyReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(sad[2], 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6),
            nn.LeakyReLU()
        )
        self.mlp4 = nn.Sequential(
            nn.Linear(sad[3], 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6),
            nn.LeakyReLU()
        )
        self.mlp5 = nn.Sequential(
            nn.Linear(sad[4], 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4),
            nn.LeakyReLU()
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(32, 32, 5, 5),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(32, 32, 5, 1),
            nn.LeakyReLU(),
            nn.Conv1d(32, 32, 5, 1),
            nn.LeakyReLU(),
        )
        if length == 10:
            self.final = nn.Linear(64 * 13 * 2, out_dim * 2)
        elif length == 20:
            self.final = nn.Linear(32 * 12, out_dim * 2)

        self.mu_layer = nn.Linear(out_dim * 2, out_dim)
        self.std_layer = nn.Linear(out_dim * 2, out_dim)

        self.conv1 = torch.nn.Conv2d(8, 32, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.pooling = torch.nn.AvgPool2d(2)

    def forward(self,bsz, piece, cam_obs_cur, cam_obs_goal, tcp_translation, tcp_rv, tcp_theta):
        length =cam_obs_cur.shape[2]

        res1 = self.mlp1(cam_obs_cur).permute(0, 1, 3, 2)#.view(bsz,1,32,length)
        res2 = self.mlp2(cam_obs_goal).permute(0, 1, 3, 2)#.view(bsz,1,32,length)
        res3 = self.mlp3(tcp_translation).permute(0, 1, 3, 2)#.view(bsz,1,32,length)
        res4 = self.mlp4(tcp_rv).permute(0, 1, 3, 2)#.view(bsz,1,32,length)
        res5 = self.mlp5(tcp_theta).permute(0, 1, 3, 2)#.view(bsz,1,32,length)

        res = torch.cat([res1,res2,res3,res4,res5],dim=2).view(bsz,piece,32,length)
        res = self.conv1(res)
        res = F.leaky_relu(res)
        res = self.pooling(res)
        res = self.conv2(res)
        if length == 10:
            res = F.leaky_relu(res).reshape(bsz, 64 * 13 * 2)
        elif length == 20:
            res = F.leaky_relu(res).reshape(bsz, 32 * 12)
        res = self.final(res)
        return self._sample_latent(res)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self.mu_layer(h_enc)
        log_sigma = self.std_layer(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(device)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * std_z  # Reparameterization trick

    def get_mean_sample(self, recent_states, k=10):
        lis = []
        res = self.mlp(recent_states).permute(0, 2, 1)
        res = self.cnn(res).reshape(-1, 32 * 3)
        h_enc = self.final(res)
        if k:
            if k == 1:
                return self._sample_latent(h_enc)
            else:
                for i in range(k):
                    lis.append(self._sample_latent(h_enc).cpu().detach().numpy())
                lis = np.array(lis)
                return torch.FloatTensor(np.mean(lis, axis=0))
        else:
            return self.mu_layer(h_enc)

    def latent_loss(self, z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class SingleModule_caz_prompt(nn.Module):
    def __init__(self):
        super(SingleModule_caz_prompt, self).__init__()
        self.l1 = nn.Linear(8+8+8, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 6)
        self.l5 = nn.Linear(128, 1)

    def forward(self, state):
        bsz = state.shape[0]
        # input = state.view(bsz,-1,)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return torch.tanh(self.l4(a)),-F.relu(self.l5(a))

class TIE_Transfer_VAE7_caz_tcp_multihead_prompt(object):
    def __init__(
            self,
            max_action,
            extr,
            intr,
            length=10
    ):
        self.basecontroller = BaseController().to(device=device, dtype=torch.float32)
        self.encoder = VAE(7, 8).to(device=device, dtype=torch.float32)
        self.transfermodule = SingleModule_caz_prompt().to(device=device, dtype=torch.float32)

        self.max_action = max_action
        self.criterion = nn.MSELoss()
        self.adaption_criterion = nn.MSELoss(reduction='sum')
        self.transfer_optimizer = torch.optim.Adam([
            {'params': self.transfermodule.parameters(), 'lr': 3e-4}
        ])

        self.env_latent = 0
        self.initial_env_latent(extr, intr)
        # self.adaptionmodule = Adaptator(8+8+6,8,length=length).to(device=device, dtype=torch.float32)
        self.adaptionmodule = Adaptator_caz_conv2d_multihead_prompt([8,8,3,3,1],8,length=length).to(device=device, dtype=torch.float32)
        self.adaption_optimizer = torch.optim.Adam([
            {'params': self.adaptionmodule.parameters(), 'lr': 3e-4}
        ])

    def load_basecontroller(self, filename):
        self.basecontroller.load_state_dict(torch.load(filename + "_actor"), strict=False)

    def load_encoder(self, filename):
        self.encoder.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    def load_transfermodule(self, filename):
        self.transfermodule.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage), strict=False)

    def load_adaptionmodule(self, filename):
        self.adaptionmodule.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage), strict=False)

    def select_action_with_basecontroller(self, errstate):
        errstate = torch.Tensor(errstate).to(device=device, dtype=torch.float32).unsqueeze(0)
        return self.max_action * self.basecontroller(errstate).cpu().data.numpy().flatten()

    def select_action_with_transfermodule(self, cam_obs_cur, cam_obs_goal, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
        env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
        vel_vec, vel_norm = self.transfermodule(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten()

    def initial_env_latent(self, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
        self.env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)
        return

    def adapt_env_latent(self, cam_obs_cur, cam_obs_goal, action_net, tcp_pose,  length=10):
        cam_obs_cur = torch.Tensor(np.array(cam_obs_cur)).to(device=device, dtype=torch.float32).view(-1, 8, length, 8
                                                                                                      ) / 320 - 1
        cam_obs_goal = torch.Tensor(np.array(cam_obs_goal)).to(device=device, dtype=torch.float32).view(-1,8, length ,8
                                                                                                        ) / 320 - 1
        action_net = torch.Tensor(np.array(action_net)).to(device=device, dtype=torch.float32)


        batch_tcp_rv = []
        batch_tcp_theta = []
        for j in range(tcp_pose.shape[0]):
            tcp_rv = []
            tcp_theta = []
            for i in range(tcp_pose.shape[1]):
                tcp_rv.append(R.from_matrix(tcp_pose[j,i, :3, :3]).as_rotvec())
                tcp_theta.append(np.linalg.norm(tcp_rv[i]))
                tcp_rv[i] = tcp_rv[i]/(tcp_theta[i] + 1e-7)
            batch_tcp_rv.append(np.array([tcp_rv]))
            batch_tcp_theta.append(np.array([tcp_theta]))

        tcp_translation = torch.Tensor(tcp_pose[:,:, :3, 3]).to(device=device, dtype=torch.float32).view(-1,8, length, 3)
        tcp_rv = torch.Tensor(np.array(batch_tcp_rv)).to(device=device, dtype=torch.float32).view(-1,8, length, 3)
        tcp_theta = torch.Tensor(np.array(batch_tcp_theta)).to(device=device, dtype=torch.float32).view(-1,8, length, 1)

        action_norm = torch.norm(action_net, dim=1, keepdim=True)
        action_vec = action_net / (action_norm + 1e-7)

        bsz = 1
        piece = 8
        self.env_latent = self.adaptionmodule(bsz,piece,cam_obs_cur, cam_obs_goal,
                                     tcp_translation, tcp_rv,
                                     tcp_theta).to(device=device, dtype=torch.float32)
        return

    def select_action_with_adaptionmodule(self, cam_obs_cur, cam_obs_goal):
        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, self.env_latent], dim=1)
        vel_vec, vel_norm = self.transfermodule(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten(), vel_vec.cpu().data.numpy().flatten(), vel_norm.cpu().data.numpy().flatten()

    def train_transfer(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        total_dir_loss = 0
        total_norm_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[0].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            gt = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8) / 320 - 1

            input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
            vel_vec, vel_norm = self.transfermodule(input)
            # Compute actor losse
            dir_loss = scaled_cosine_similarity_loss(vel_vec, gt).mean()
            gt_norm = torch.norm(gt, dim=-1, keepdim=True)
            norm_loss = F.mse_loss(
                vel_norm,
                reverse_alpha_p_elu(gt_norm, eps=1e-5),
                reduction="mean"
            ) * 0.5
            actor_loss = dir_loss + norm_loss
            # Optimize the actor
            self.transfer_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.transfer_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            total_dir_loss += dir_loss
            total_norm_loss += norm_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
            tb.add_scalar('average_dir_loss', total_dir_loss / count, episode_timesteps)
            tb.add_scalar('average_norm_loss', total_norm_loss / count, episode_timesteps)
        return

    def save_transfer(self, filename):
        torch.save(self.transfermodule.state_dict(), filename)

    def train_adaption(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        total_reg_loss = 0
        total_dir_loss = 0
        total_norm_loss = 0

        count = 0
        for batch in train_loader:
            bsz = batch[1].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8, 10, 8) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8, 10, 8) / 320 - 1

            tcp_rv = batch[11].to(device=device, dtype=torch.float32).view(bsz, 8, 10, 3)
            tcp_theta = batch[12].to(device=device, dtype=torch.float32).view(bsz, 8, 10, 1)
            tcp_translation = batch[13].to(device=device, dtype=torch.float32).view(bsz, 8, 10, 3)

            # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0,2,1)
            output = self.adaptionmodule(bsz,8,cam_obs_cur, cam_obs_goal, tcp_translation, tcp_rv, tcp_theta)
            # Compute actor loss
            reg_loss = self.criterion(env_latent, output)
            # ll = self.adaptionmodule.latent_loss(self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma)
            # pre_mu_std = torch.cat((self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma), axis=1)
            # true_mu_std = torch.cat((self.encoder.z_mean, self.encoder.z_sigma), axis=1)
            # ml = self.adaption_criterion(pre_mu_std, true_mu_std)
            # actor_loss = ml
            # actor_loss = ll + ml
            # Optimize the actor

            traj_gt = batch[14].to(device=device, dtype=torch.float32)
            traj_cam_obs_cur = batch[15].to(device=device, dtype=torch.float32).view(bsz, 20, 8) / 320 - 1
            traj_cam_obs_goal = batch[16].to(device=device, dtype=torch.float32).view(bsz, 20, 8) / 320 - 1

            list_latent = []
            for i in range(bsz):
                list_latent.append(output[i].repeat(20,1).unsqueeze(0))
            cal_latent = torch.cat(list_latent,dim = 0)

            input = torch.cat([traj_cam_obs_cur, traj_cam_obs_goal, cal_latent], dim=2)
            vel_vec, vel_norm = self.transfermodule(input)
            # Compute actor losse
            dir_loss = scaled_cosine_similarity_loss(vel_vec, traj_gt).mean()
            gt_norm = torch.norm(traj_gt, dim=-1, keepdim=True)
            norm_loss = F.mse_loss(
                vel_norm,
                reverse_alpha_p_elu(gt_norm, eps=1e-5),
                reduction="mean"
            )

            actor_loss = reg_loss + dir_loss + norm_loss

            self.adaption_optimizer.zero_grad()

            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            total_reg_loss += reg_loss
            total_dir_loss += dir_loss
            total_norm_loss += norm_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
            tb.add_scalar('average_reg_loss', total_reg_loss / count, episode_timesteps)
            tb.add_scalar('average_dir_loss', total_dir_loss / count, episode_timesteps)
            tb.add_scalar('average_norm_loss', total_norm_loss / count, episode_timesteps)
        return

    def save_adaption(self, filename):
        torch.save(self.adaptionmodule.state_dict(), filename)

class TIE_Transfer_VAE7_caz_tcp_velloss(object):
    def __init__(
            self,
            max_action,
            extr,
            intr,
            length=10
    ):
        self.basecontroller = BaseController().to(device=device, dtype=torch.float32)
        self.encoder = VAE(7, 8).to(device=device, dtype=torch.float32)
        self.transfermodule = SingleModule_caz_prompt().to(device=device, dtype=torch.float32)

        self.max_action = max_action
        self.criterion = nn.MSELoss()
        self.adaption_criterion = nn.MSELoss(reduction='sum')
        self.transfer_optimizer = torch.optim.Adam([
            {'params': self.transfermodule.parameters(), 'lr': 3e-4}
        ])

        self.env_latent = 0
        self.initial_env_latent(extr, intr)
        # self.adaptionmodule = Adaptator(8+8+6,8,length=length).to(device=device, dtype=torch.float32)
        self.adaptionmodule = Adaptator_caz_conv2d(8+8+7+7,8,length=length).to(device=device, dtype=torch.float32)
        self.adaption_optimizer = torch.optim.Adam([
            {'params': self.adaptionmodule.parameters(), 'lr': 3e-4}
        ])

    def load_basecontroller(self, filename):
        self.basecontroller.load_state_dict(torch.load(filename + "_actor"), strict=False)

    def load_encoder(self, filename):
        self.encoder.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    def load_transfermodule(self, filename):
        self.transfermodule.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage), strict=False)

    def load_adaptionmodule(self, filename):
        self.adaptionmodule.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage), strict=False)

    def select_action_with_basecontroller(self, errstate):
        errstate = torch.Tensor(errstate).to(device=device, dtype=torch.float32).unsqueeze(0)
        return self.max_action * self.basecontroller(errstate).cpu().data.numpy().flatten()

    def select_action_with_transfermodule(self, cam_obs_cur, cam_obs_goal, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
        env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
        vel_vec, vel_norm = self.transfermodule(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten()

    def initial_env_latent(self, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
        self.env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)
        return

    def adapt_env_latent(self, cam_obs_cur, cam_obs_goal, action_net, tcp_pose, last_latent,  length=10):
        cam_obs_cur = torch.Tensor(np.array(cam_obs_cur)).to(device=device, dtype=torch.float32).view(1, 8,
                                                                                                      length) / 320 - 1
        cam_obs_goal = torch.Tensor(np.array(cam_obs_goal)).to(device=device, dtype=torch.float32).view(1, 8,
                                                                                                        length) / 320 - 1
        action_net = torch.Tensor(np.array(action_net)).to(device=device, dtype=torch.float32)

        tcp_rv = []
        tcp_theta = []
        for i in range(tcp_pose.shape[0]):
            tcp_rv.append(R.from_matrix(tcp_pose[i, :3, :3]).as_rotvec())
            tcp_theta.append(np.linalg.norm(tcp_rv[i]))
            tcp_rv[i] = tcp_rv[i]/(tcp_theta[i] + 1e-7)
        tcp_translation = torch.Tensor(tcp_pose[:, :3, 3]).to(device=device, dtype=torch.float32).view(1, 3, length)
        tcp_rv = torch.Tensor(np.array(tcp_rv)).to(device=device, dtype=torch.float32).view(1, 3, length)
        tcp_theta = torch.Tensor(np.array(tcp_theta)).to(device=device, dtype=torch.float32).view(1, 1, length)

        last_latent = torch.Tensor(np.array(last_latent)).to(device=device, dtype=torch.float32).view(1,8)
        # input = torch.cat([cam_obs_cur, cam_obs_goal, action_net], dim=1).permute(0,2,1)
        # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0, 2, 1)
        action_norm = torch.norm(action_net, dim=1, keepdim=True)
        action_vec = action_net / (action_norm + 1e-7)

        input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(1, 6,length),
                           action_norm.view(1, 1,length),tcp_translation,tcp_rv,tcp_theta], dim=1).permute(0, 2, 1)
        self.env_latent = self.adaptionmodule(input).to(device=device, dtype=torch.float32)
        return

    def select_action_with_adaptionmodule(self, cam_obs_cur, cam_obs_goal):
        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, 8) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, self.env_latent], dim=1)
        vel_vec, vel_norm = self.transfermodule(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten(), vel_vec.cpu().data.numpy().flatten(), vel_norm.cpu().data.numpy().flatten()

    def train_transfer(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        total_dir_loss = 0
        total_norm_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[0].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            gt = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8) / 320 - 1

            input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
            vel_vec, vel_norm = self.transfermodule(input)
            # Compute actor losse
            dir_loss = scaled_cosine_similarity_loss(vel_vec, gt).mean()
            gt_norm = torch.norm(gt, dim=-1, keepdim=True)
            norm_loss = F.mse_loss(
                vel_norm,
                reverse_alpha_p_elu(gt_norm, eps=1e-5),
                reduction="mean"
            ) * 0.5
            actor_loss = dir_loss + norm_loss
            # Optimize the actor
            self.transfer_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.transfer_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            total_dir_loss += dir_loss
            total_norm_loss += norm_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
            tb.add_scalar('average_dir_loss', total_dir_loss / count, episode_timesteps)
            tb.add_scalar('average_norm_loss', total_norm_loss / count, episode_timesteps)
        return

    def save_transfer(self, filename):
        torch.save(self.transfermodule.state_dict(), filename)

    def train_adaption(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[1].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32)
            last_latent = batch[10].to(device=device, dtype=torch.float32).view(bsz, 8)

            tcp_rv = batch[11].to(device=device, dtype=torch.float32).view(bsz, 3, -1)
            tcp_theta = batch[12].to(device=device, dtype=torch.float32).unsqueeze(1)
            tcp_translation = batch[13].to(device=device, dtype=torch.float32).view(bsz, 3, -1)

            action_norm = torch.norm(action_net, dim=-1, keepdim=True)
            action_vec = action_net / (action_norm + 1e-7)

            # input = torch.cat(
            #     [cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1), action_norm.view(bsz, 1, -1)],
            #     dim=1).permute(0, 2, 1)
            input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1),
                               action_norm.view(bsz, 1, -1), tcp_translation, tcp_rv, tcp_theta], dim=1).permute(0, 2,
                                                                                                                   1)



            # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0,2,1)
            output = self.adaptionmodule(input)
            # Compute actor loss
            actor_loss = self.criterion(env_latent, output)
            # ll = self.adaptionmodule.latent_loss(self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma)
            # pre_mu_std = torch.cat((self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma), axis=1)
            # true_mu_std = torch.cat((self.encoder.z_mean, self.encoder.z_sigma), axis=1)
            # ml = self.adaption_criterion(pre_mu_std, true_mu_std)
            # actor_loss = ml
            # actor_loss = ll + ml
            # Optimize the actor
            self.adaption_optimizer.zero_grad()

            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def train_adaption_velloss(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        total_reg_loss = 0
        total_dir_loss = 0
        total_norm_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[1].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            gt = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 8, -1) / 320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32)
            last_latent = batch[10].to(device=device, dtype=torch.float32).view(bsz, 8)

            tcp_rv = batch[11].to(device=device, dtype=torch.float32).view(bsz, 3, -1)
            tcp_theta = batch[12].to(device=device, dtype=torch.float32).unsqueeze(1)
            tcp_translation = batch[13].to(device=device, dtype=torch.float32).view(bsz, 3, -1)

            action_norm = torch.norm(action_net, dim=-1, keepdim=True)
            action_vec = action_net / (action_norm + 1e-7)

            # input = torch.cat(
            #     [cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1), action_norm.view(bsz, 1, -1)],
            #     dim=1).permute(0, 2, 1)
            input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1),
                               action_norm.view(bsz, 1, -1), tcp_translation, tcp_rv, tcp_theta], dim=1).permute(0, 2,
                                                                                                                   1)

            # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0,2,1)
            output = self.adaptionmodule(input)
            # Compute actor loss
            reg_loss = self.criterion(env_latent, output)

            # ll = self.adaptionmodule.latent_loss(self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma)
            # pre_mu_std = torch.cat((self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma), axis=1)
            # true_mu_std = torch.cat((self.encoder.z_mean, self.encoder.z_sigma), axis=1)
            # ml = self.adaption_criterion(pre_mu_std, true_mu_std)
            # actor_loss = ml
            # actor_loss = ll + ml
            # Optimize the actor

            traj_gt = batch[6].to(device=device, dtype=torch.float32)
            traj_cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 10, 8) / 320 - 1
            traj_cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 10, 8) / 320 - 1
            list_latent = []
            for i in range(bsz):
                list_latent.append(output[i].repeat(10,1).unsqueeze(0))
            cal_latent = torch.cat(list_latent,dim = 0)
            input = torch.cat([traj_cam_obs_cur, traj_cam_obs_goal, cal_latent], dim=2)
            vel_vec, vel_norm = self.transfermodule(input)
            # Compute actor losse
            dir_loss = scaled_cosine_similarity_loss(vel_vec, traj_gt).mean()
            gt_norm = torch.norm(traj_gt, dim=-1, keepdim=True)
            norm_loss = F.mse_loss(
                vel_norm,
                reverse_alpha_p_elu(gt_norm, eps=1e-5),
                reduction="mean"
            ) * 0.5
            actor_loss = dir_loss + norm_loss + reg_loss

            self.adaption_optimizer.zero_grad()

            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            total_reg_loss += reg_loss
            total_dir_loss += dir_loss
            total_norm_loss += norm_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
            tb.add_scalar('average_reg_loss', total_reg_loss / count, episode_timesteps)
            tb.add_scalar('average_dir_loss', total_dir_loss / count, episode_timesteps)
            tb.add_scalar('average_norm_loss', total_norm_loss / count, episode_timesteps)
        return

    def save_adaption(self, filename):
        torch.save(self.adaptionmodule.state_dict(), filename)


class SingleModule_caz_prompt_general(nn.Module):
    def __init__(self,input_dim=24):
        super(SingleModule_caz_prompt_general, self).__init__()
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 6)
        self.l5 = nn.Linear(128, 1)

    def forward(self, state):
        bsz = state.shape[0]
        # input = state.view(bsz,-1,)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return torch.tanh(self.l4(a)),-F.relu(self.l5(a))

class TIE_Transfer_VAE7_caz_tcp_velloss_general(object):
    def __init__(
            self,
            max_action,
            extr,
            intr,
            input_dim=8,
            length=10,
            observer = None
    ):
        self.input_dim = input_dim
        self.basecontroller = BasePolicy(2*input_dim).to(device=device, dtype=torch.float32)
        self.encoder = VAE(7, 8).to(device=device, dtype=torch.float32)
        self.transfermodule = SingleModule_caz_prompt_general(2*input_dim+8).to(device=device, dtype=torch.float32)

        self.max_action = max_action
        self.criterion = nn.MSELoss()
        self.adaption_criterion = nn.MSELoss(reduction='sum')
        self.transfer_optimizer = torch.optim.Adam([
            {'params': self.transfermodule.parameters(), 'lr': 3e-4}
        ])
        self.base_optimizer = torch.optim.Adam([
            {'params': self.basecontroller.parameters(), 'lr': 3e-4}
        ])

        self.env_latent = 0
        self.initial_env_latent(extr, intr)
        # self.adaptionmodule = Adaptator(8+8+6,8,length=length).to(device=device, dtype=torch.float32)
        self.adaptionmodule = Adaptator_caz_conv2d(2*input_dim+7+7,8,length=length).to(device=device, dtype=torch.float32)
        self.adaption_optimizer = torch.optim.Adam([
            {'params': self.adaptionmodule.parameters(), 'lr': 3e-4}
        ])
        if observer == 'apriltag':
            self.observer = Apriltag()
        elif observer == 'charge':
            self.observer = RES()
            filename = '/home/shahao/yhx/simple-pybullet-ur5-modeldatesetlog/models/TD3_vs_gym:charge-v1_10_220717_0037_charge_real_ob300_fewshot_coe_resnetdagger3/TD3_vs_gym:charge-v1_10_220717_0037_charge_real_ob300_fewshot_coe_resnetdagger3_660'
            self.observer.net.load_state_dict(torch.load(filename + "_observer", map_location=device))
        elif observer == 'horse':
            self.observer = RES_horse()

    def load_basecontroller(self, filename):
        self.basecontroller.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    def load_encoder(self, filename):
        self.encoder.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    def load_transfermodule(self, filename):
        self.transfermodule.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage), strict=False)

    def load_adaptionmodule(self, filename):
        self.adaptionmodule.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage), strict=False)

    def select_action_with_basecontroller(self, cam_obs_cur, cam_obs_goal):
        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, self.input_dim) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, self.input_dim) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal], dim=1)
        vel_vec, vel_norm = self.basecontroller(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten()

    def select_action_with_transfermodule(self, cam_obs_cur, cam_obs_goal, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
        env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, self.input_dim) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, self.input_dim) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
        vel_vec, vel_norm = self.transfermodule(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten()

    def initial_env_latent(self, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
        self.env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)
        return

    def adapt_env_latent(self, cam_obs_cur, cam_obs_goal, action_net, tcp_pose, last_latent,  length=10):
        cam_obs_cur = torch.Tensor(np.array(cam_obs_cur)).to(device=device, dtype=torch.float32).view(1, self.input_dim,
                                                                                                      length) / 320 - 1
        cam_obs_goal = torch.Tensor(np.array(cam_obs_goal)).to(device=device, dtype=torch.float32).view(1, self.input_dim,
                                                                                                        length) / 320 - 1
        action_net = torch.Tensor(np.array(action_net)).to(device=device, dtype=torch.float32)

        tcp_rv = []
        tcp_theta = []
        for i in range(tcp_pose.shape[0]):
            tcp_rv.append(R.from_matrix(tcp_pose[i, :3, :3]).as_rotvec())
            tcp_theta.append(np.linalg.norm(tcp_rv[i]))
            tcp_rv[i] = tcp_rv[i]/(tcp_theta[i] + 1e-7)
        tcp_translation = torch.Tensor(tcp_pose[:, :3, 3]).to(device=device, dtype=torch.float32).view(1, 3, length)
        tcp_rv = torch.Tensor(np.array(tcp_rv)).to(device=device, dtype=torch.float32).view(1, 3, length)
        tcp_theta = torch.Tensor(np.array(tcp_theta)).to(device=device, dtype=torch.float32).view(1, 1, length)

        last_latent = torch.Tensor(np.array(last_latent)).to(device=device, dtype=torch.float32).view(1,8)
        # input = torch.cat([cam_obs_cur, cam_obs_goal, action_net], dim=1).permute(0,2,1)
        # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0, 2, 1)
        action_norm = torch.norm(action_net, dim=1, keepdim=True)
        action_vec = action_net / (action_norm + 1e-7)

        input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(1, 6,length),
                           action_norm.view(1, 1,length),tcp_translation,tcp_rv,tcp_theta], dim=1).permute(0, 2, 1)
        self.env_latent = self.adaptionmodule(input).to(device=device, dtype=torch.float32)
        return

    def select_action_with_adaptionmodule(self, cam_obs_cur, cam_obs_goal):
        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, self.input_dim) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, self.input_dim) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, self.env_latent], dim=1)
        vel_vec, vel_norm = self.transfermodule(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten(), vel_vec.cpu().data.numpy().flatten(), vel_norm.cpu().data.numpy().flatten()

    def train_transfer(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        total_dir_loss = 0
        total_norm_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[0].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            gt = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, self.input_dim) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, self.input_dim) / 320 - 1

            input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
            vel_vec, vel_norm = self.transfermodule(input)
            # Compute actor losse
            dir_loss = scaled_cosine_similarity_loss(vel_vec, gt).mean()
            gt_norm = torch.norm(gt, dim=-1, keepdim=True)
            norm_loss = F.mse_loss(
                vel_norm,
                reverse_alpha_p_elu(gt_norm, eps=1e-5),
                reduction="mean"
            ) * 0.5
            actor_loss = dir_loss + norm_loss
            # Optimize the actor
            self.transfer_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.transfer_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            total_dir_loss += dir_loss
            total_norm_loss += norm_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
            tb.add_scalar('average_dir_loss', total_dir_loss / count, episode_timesteps)
            tb.add_scalar('average_norm_loss', total_norm_loss / count, episode_timesteps)
        return

    def train_base(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        total_dir_loss = 0
        total_norm_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[0].shape[0]

            gt = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, self.input_dim) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, self.input_dim) / 320 - 1

            input = torch.cat([cam_obs_cur, cam_obs_goal], dim=1)
            vel_vec, vel_norm = self.basecontroller(input)
            # Compute actor losse
            dir_loss = scaled_cosine_similarity_loss(vel_vec, gt).mean()
            gt_norm = torch.norm(gt, dim=-1, keepdim=True)
            norm_loss = F.mse_loss(
                vel_norm,
                reverse_alpha_p_elu(gt_norm, eps=1e-5),
                reduction="mean"
            ) * 0.5
            actor_loss = dir_loss + norm_loss
            # Optimize the actor
            self.base_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.base_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            total_dir_loss += dir_loss
            total_norm_loss += norm_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
            tb.add_scalar('average_dir_loss', total_dir_loss / count, episode_timesteps)
            tb.add_scalar('average_norm_loss', total_norm_loss / count, episode_timesteps)
        return

    def save_transfer(self, filename):
        torch.save(self.transfermodule.state_dict(), filename)

    def save_base(self, filename):
        torch.save(self.basecontroller.state_dict(), filename)

    def train_adaption(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[1].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, self.input_dim, -1) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, self.input_dim, -1) / 320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32)
            last_latent = batch[10].to(device=device, dtype=torch.float32).view(bsz, 8)

            tcp_rv = batch[11].to(device=device, dtype=torch.float32).view(bsz, 3, -1)
            tcp_theta = batch[12].to(device=device, dtype=torch.float32).unsqueeze(1)
            tcp_translation = batch[13].to(device=device, dtype=torch.float32).view(bsz, 3, -1)

            action_norm = torch.norm(action_net, dim=-1, keepdim=True)
            action_vec = action_net / (action_norm + 1e-7)

            # input = torch.cat(
            #     [cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1), action_norm.view(bsz, 1, -1)],
            #     dim=1).permute(0, 2, 1)
            input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1),
                               action_norm.view(bsz, 1, -1), tcp_translation, tcp_rv, tcp_theta], dim=1).permute(0, 2,
                                                                                                                   1)



            # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0,2,1)
            output = self.adaptionmodule(input)
            # Compute actor loss
            actor_loss = self.criterion(env_latent, output)
            # ll = self.adaptionmodule.latent_loss(self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma)
            # pre_mu_std = torch.cat((self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma), axis=1)
            # true_mu_std = torch.cat((self.encoder.z_mean, self.encoder.z_sigma), axis=1)
            # ml = self.adaption_criterion(pre_mu_std, true_mu_std)
            # actor_loss = ml
            # actor_loss = ll + ml
            # Optimize the actor
            self.adaption_optimizer.zero_grad()

            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def train_adaption_velloss(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        total_reg_loss = 0
        total_dir_loss = 0
        total_norm_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[1].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            gt = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, self.input_dim, -1) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, self.input_dim, -1) / 320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32)
            last_latent = batch[10].to(device=device, dtype=torch.float32).view(bsz, 8)

            tcp_rv = batch[11].to(device=device, dtype=torch.float32).view(bsz, 3, -1)
            tcp_theta = batch[12].to(device=device, dtype=torch.float32).unsqueeze(1)
            tcp_translation = batch[13].to(device=device, dtype=torch.float32).view(bsz, 3, -1)

            action_norm = torch.norm(action_net, dim=-1, keepdim=True)
            action_vec = action_net / (action_norm + 1e-7)

            # input = torch.cat(
            #     [cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1), action_norm.view(bsz, 1, -1)],
            #     dim=1).permute(0, 2, 1)
            input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1),
                               action_norm.view(bsz, 1, -1), tcp_translation, tcp_rv, tcp_theta], dim=1).permute(0, 2,
                                                                                                                   1)

            # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0,2,1)
            output = self.adaptionmodule(input)
            # Compute actor loss
            reg_loss = self.criterion(env_latent, output)

            # ll = self.adaptionmodule.latent_loss(self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma)
            # pre_mu_std = torch.cat((self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma), axis=1)
            # true_mu_std = torch.cat((self.encoder.z_mean, self.encoder.z_sigma), axis=1)
            # ml = self.adaption_criterion(pre_mu_std, true_mu_std)
            # actor_loss = ml
            # actor_loss = ll + ml
            # Optimize the actor

            traj_gt = batch[6].to(device=device, dtype=torch.float32)
            traj_cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 10, self.input_dim) / 320 - 1
            traj_cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 10, self.input_dim) / 320 - 1
            list_latent = []
            for i in range(bsz):
                list_latent.append(output[i].repeat(10,1).unsqueeze(0))
            cal_latent = torch.cat(list_latent,dim = 0)
            input = torch.cat([traj_cam_obs_cur, traj_cam_obs_goal, cal_latent], dim=2)
            vel_vec, vel_norm = self.transfermodule(input)
            # Compute actor losse
            dir_loss = scaled_cosine_similarity_loss(vel_vec, traj_gt).mean()
            gt_norm = torch.norm(traj_gt, dim=-1, keepdim=True)
            norm_loss = F.mse_loss(
                vel_norm,
                reverse_alpha_p_elu(gt_norm, eps=1e-5),
                reduction="mean"
            ) * 0.5
            actor_loss = dir_loss + norm_loss + reg_loss

            self.adaption_optimizer.zero_grad()

            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            total_reg_loss += reg_loss
            total_dir_loss += dir_loss
            total_norm_loss += norm_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
            tb.add_scalar('average_reg_loss', total_reg_loss / count, episode_timesteps)
            tb.add_scalar('average_dir_loss', total_dir_loss / count, episode_timesteps)
            tb.add_scalar('average_norm_loss', total_norm_loss / count, episode_timesteps)
        return

    def train_adaption_regloss(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        total_reg_loss = 0
        total_dir_loss = 0
        total_norm_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[1].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            gt = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, self.input_dim, -1) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, self.input_dim, -1) / 320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32)
            last_latent = batch[10].to(device=device, dtype=torch.float32).view(bsz, 8)

            tcp_rv = batch[11].to(device=device, dtype=torch.float32).view(bsz, 3, -1)
            tcp_theta = batch[12].to(device=device, dtype=torch.float32).unsqueeze(1)
            tcp_translation = batch[13].to(device=device, dtype=torch.float32).view(bsz, 3, -1)

            action_norm = torch.norm(action_net, dim=-1, keepdim=True)
            action_vec = action_net / (action_norm + 1e-7)

            # input = torch.cat(
            #     [cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1), action_norm.view(bsz, 1, -1)],
            #     dim=1).permute(0, 2, 1)
            input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1),
                               action_norm.view(bsz, 1, -1), tcp_translation, tcp_rv, tcp_theta], dim=1).permute(0, 2,
                                                                                                                   1)

            # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0,2,1)
            output = self.adaptionmodule(input)
            # Compute actor loss
            reg_loss = self.criterion(env_latent, output)

            # ll = self.adaptionmodule.latent_loss(self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma)
            # pre_mu_std = torch.cat((self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma), axis=1)
            # true_mu_std = torch.cat((self.encoder.z_mean, self.encoder.z_sigma), axis=1)
            # ml = self.adaption_criterion(pre_mu_std, true_mu_std)
            # actor_loss = ml
            # actor_loss = ll + ml
            # Optimize the actor

            traj_gt = batch[6].to(device=device, dtype=torch.float32)
            traj_cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 10, self.input_dim) / 320 - 1
            traj_cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 10, self.input_dim) / 320 - 1
            list_latent = []
            for i in range(bsz):
                list_latent.append(output[i].repeat(10,1).unsqueeze(0))
            cal_latent = torch.cat(list_latent,dim = 0)
            input = torch.cat([traj_cam_obs_cur, traj_cam_obs_goal, cal_latent], dim=2)
            vel_vec, vel_norm = self.transfermodule(input)
            # Compute actor losse
            dir_loss = scaled_cosine_similarity_loss(vel_vec, traj_gt).mean()
            gt_norm = torch.norm(traj_gt, dim=-1, keepdim=True)
            norm_loss = F.mse_loss(
                vel_norm,
                reverse_alpha_p_elu(gt_norm, eps=1e-5),
                reduction="mean"
            ) * 0.5
            actor_loss = reg_loss

            self.adaption_optimizer.zero_grad()

            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            total_reg_loss += reg_loss
            total_dir_loss += dir_loss
            total_norm_loss += norm_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
            tb.add_scalar('average_reg_loss', total_reg_loss / count, episode_timesteps)
            tb.add_scalar('average_dir_loss', total_dir_loss / count, episode_timesteps)
            tb.add_scalar('average_norm_loss', total_norm_loss / count, episode_timesteps)
        return

    def save_adaption(self, filename):
        torch.save(self.adaptionmodule.state_dict(), filename)

class BasePolicy(nn.Module):
    def __init__(self,input_dim):
        super(BasePolicy, self).__init__()
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 6)
        self.l5 = nn.Linear(128, 1)

    def forward(self, state):
        bsz = state.shape[0]
        # input = state.view(bsz,-1,)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return torch.tanh(self.l4(a)), -F.relu(self.l5(a))

class TIE_Transfer_VAE10_caz_tcp_velloss_general(object):
    def __init__(
            self,
            max_action,
            extr,
            intr,
            input_dim=8,
            length=10
    ):
        self.input_dim = input_dim
        self.basecontroller = BasePolicy(2*input_dim).to(device=device, dtype=torch.float32)
        self.encoder = VAE(10, 10).to(device=device, dtype=torch.float32)
        self.transfermodule = SingleModule_caz_prompt_general(2*input_dim+10).to(device=device, dtype=torch.float32)

        self.max_action = max_action
        self.criterion = nn.MSELoss()
        self.adaption_criterion = nn.MSELoss(reduction='sum')
        self.transfer_optimizer = torch.optim.Adam([
            {'params': self.transfermodule.parameters(), 'lr': 3e-4}
        ])

        self.env_latent = 0
        self.initial_env_latent(extr, intr)
        # self.adaptionmodule = Adaptator(8+8+6,8,length=length).to(device=device, dtype=torch.float32)
        self.adaptionmodule = Adaptator_caz_conv2d(2*input_dim+7+7,10,length=length).to(device=device, dtype=torch.float32)
        self.adaption_optimizer = torch.optim.Adam([
            {'params': self.adaptionmodule.parameters(), 'lr': 3e-4}
        ])

    def load_basecontroller(self, filename):
        self.basecontroller.load_state_dict(torch.load(filename + "_actor"), strict=False)

    def load_encoder(self, filename):
        self.encoder.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    def load_transfermodule(self, filename):
        self.transfermodule.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage), strict=False)

    def load_adaptionmodule(self, filename):
        self.adaptionmodule.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage), strict=False)

    def select_action_with_basecontroller(self, errstate):
        errstate = torch.Tensor(errstate).to(device=device, dtype=torch.float32).unsqueeze(0)
        return self.max_action * self.basecontroller(errstate).cpu().data.numpy().flatten()

    def select_action_with_transfermodule(self, cam_obs_cur, cam_obs_goal, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([translation, rv, theta, f, cx, cy], dim=1)
        env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, self.input_dim) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, self.input_dim) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
        vel_vec, vel_norm = self.transfermodule(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten()

    def initial_env_latent(self, extr, intr):
        translation = torch.Tensor(extr[:3, 3]).to(device=device, dtype=torch.float32).view(1, 3)
        rv = R.from_matrix(extr[:3, :3]).as_rotvec()
        theta = np.linalg.norm(rv)
        rv = rv / theta
        theta = theta / np.pi
        f, cx, cy = (intr[0, 0] - 610) / 30, (intr[0, 2] - 320) / 30, (intr[1, 2] - 240) / 30
        rv = torch.Tensor(rv).to(device=device, dtype=torch.float32).unsqueeze(0)
        theta = torch.Tensor(np.array(theta)).to(device=device, dtype=torch.float32).view(1, 1)
        f = torch.Tensor(np.array(f)).to(device=device, dtype=torch.float32).view(1, 1)
        cx = torch.Tensor(np.array(cx)).to(device=device, dtype=torch.float32).view(1, 1)
        cy = torch.Tensor(np.array(cy)).to(device=device, dtype=torch.float32).view(1, 1)
        env_parameters = torch.cat([translation, rv, theta, f, cx, cy], dim=1)
        self.env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)
        return

    def adapt_env_latent(self, cam_obs_cur, cam_obs_goal, action_net, tcp_pose, last_latent,  length=10):
        cam_obs_cur = torch.Tensor(np.array(cam_obs_cur)).to(device=device, dtype=torch.float32).view(1, self.input_dim,
                                                                                                      length) / 320 - 1
        cam_obs_goal = torch.Tensor(np.array(cam_obs_goal)).to(device=device, dtype=torch.float32).view(1, self.input_dim,
                                                                                                        length) / 320 - 1
        action_net = torch.Tensor(np.array(action_net)).to(device=device, dtype=torch.float32)

        tcp_rv = []
        tcp_theta = []
        for i in range(tcp_pose.shape[0]):
            tcp_rv.append(R.from_matrix(tcp_pose[i, :3, :3]).as_rotvec())
            tcp_theta.append(np.linalg.norm(tcp_rv[i]))
            tcp_rv[i] = tcp_rv[i]/(tcp_theta[i] + 1e-7)
        tcp_translation = torch.Tensor(tcp_pose[:, :3, 3]).to(device=device, dtype=torch.float32).view(1, 3, length)
        tcp_rv = torch.Tensor(np.array(tcp_rv)).to(device=device, dtype=torch.float32).view(1, 3, length)
        tcp_theta = torch.Tensor(np.array(tcp_theta)).to(device=device, dtype=torch.float32).view(1, 1, length)

        last_latent = torch.Tensor(np.array(last_latent)).to(device=device, dtype=torch.float32).view(1,10)
        # input = torch.cat([cam_obs_cur, cam_obs_goal, action_net], dim=1).permute(0,2,1)
        # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0, 2, 1)
        action_norm = torch.norm(action_net, dim=1, keepdim=True)
        action_vec = action_net / (action_norm + 1e-7)

        input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(1, 6,length),
                           action_norm.view(1, 1,length),tcp_translation,tcp_rv,tcp_theta], dim=1).permute(0, 2, 1)
        self.env_latent = self.adaptionmodule(input).to(device=device, dtype=torch.float32)
        return

    def select_action_with_adaptionmodule(self, cam_obs_cur, cam_obs_goal):
        cam_obs_cur = torch.Tensor(cam_obs_cur).to(device=device, dtype=torch.float32).view(1, self.input_dim) / 320 - 1
        cam_obs_goal = torch.Tensor(cam_obs_goal).to(device=device, dtype=torch.float32).view(1, self.input_dim) / 320 - 1

        input = torch.cat([cam_obs_cur, cam_obs_goal, self.env_latent], dim=1)
        vel_vec, vel_norm = self.transfermodule(input)

        vel = vel_vec / (torch.norm(
            vel_vec, dim=-1, keepdim=True) + 1e-7) * \
            alpha_p_elu(vel_norm, alpha=1)

        return vel.cpu().data.numpy().flatten(), vel_vec.cpu().data.numpy().flatten(), vel_norm.cpu().data.numpy().flatten()

    def train_transfer(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        total_dir_loss = 0
        total_norm_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[0].shape[0]
            translation = batch[0].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([translation, rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            gt = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, self.input_dim) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, self.input_dim) / 320 - 1

            input = torch.cat([cam_obs_cur, cam_obs_goal, env_latent], dim=1)
            vel_vec, vel_norm = self.transfermodule(input)
            # Compute actor losse
            dir_loss = scaled_cosine_similarity_loss(vel_vec, gt).mean()
            gt_norm = torch.norm(gt, dim=-1, keepdim=True)
            norm_loss = F.mse_loss(
                vel_norm,
                reverse_alpha_p_elu(gt_norm, eps=1e-5),
                reduction="mean"
            ) * 0.5
            actor_loss = dir_loss + norm_loss
            # Optimize the actor
            self.transfer_optimizer.zero_grad()
            # self.controller_optimizer.zero_grad()
            actor_loss.backward()
            self.transfer_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            total_dir_loss += dir_loss
            total_norm_loss += norm_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
            tb.add_scalar('average_dir_loss', total_dir_loss / count, episode_timesteps)
            tb.add_scalar('average_norm_loss', total_norm_loss / count, episode_timesteps)
        return

    def save_transfer(self, filename):
        torch.save(self.transfermodule.state_dict(), filename)

    def train_adaption(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[1].shape[0]
            translation = batch[0].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([translation, rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            action_pbvs = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, self.input_dim, -1) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, self.input_dim, -1) / 320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32)
            last_latent = batch[10].to(device=device, dtype=torch.float32).view(bsz, 8)

            tcp_rv = batch[11].to(device=device, dtype=torch.float32).view(bsz, 3, -1)
            tcp_theta = batch[12].to(device=device, dtype=torch.float32).unsqueeze(1)
            tcp_translation = batch[13].to(device=device, dtype=torch.float32).view(bsz, 3, -1)

            action_norm = torch.norm(action_net, dim=-1, keepdim=True)
            action_vec = action_net / (action_norm + 1e-7)

            # input = torch.cat(
            #     [cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1), action_norm.view(bsz, 1, -1)],
            #     dim=1).permute(0, 2, 1)
            input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1),
                               action_norm.view(bsz, 1, -1), tcp_translation, tcp_rv, tcp_theta], dim=1).permute(0, 2,
                                                                                                                   1)



            # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0,2,1)
            output = self.adaptionmodule(input)
            # Compute actor loss
            actor_loss = self.criterion(env_latent, output)
            # ll = self.adaptionmodule.latent_loss(self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma)
            # pre_mu_std = torch.cat((self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma), axis=1)
            # true_mu_std = torch.cat((self.encoder.z_mean, self.encoder.z_sigma), axis=1)
            # ml = self.adaption_criterion(pre_mu_std, true_mu_std)
            # actor_loss = ml
            # actor_loss = ll + ml
            # Optimize the actor
            self.adaption_optimizer.zero_grad()

            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
        return

    def train_adaption_velloss(self, train_loader, batch_size=32, tb=None, episode_timesteps=None):
        dataset_size = train_loader.__len__()
        total_actor_loss = 0
        total_reg_loss = 0
        total_dir_loss = 0
        total_norm_loss = 0
        count = 0
        for batch in train_loader:
            bsz = batch[1].shape[0]
            translation = batch[0][:, :3, 3].to(device=device, dtype=torch.float32)
            rv = batch[1].to(device=device, dtype=torch.float32)
            theta = batch[2].to(device=device, dtype=torch.float32).unsqueeze(1)
            f = batch[3].to(device=device, dtype=torch.float32).unsqueeze(1)
            cx = batch[4].to(device=device, dtype=torch.float32).unsqueeze(1)
            cy = batch[5].to(device=device, dtype=torch.float32).unsqueeze(1)
            env_parameters = torch.cat([ rv, theta, f, cx, cy], dim=1)
            env_latent = self.encoder.get_mean_sample(env_parameters).to(device=device, dtype=torch.float32)

            gt = batch[6].to(device=device, dtype=torch.float32)
            cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, self.input_dim, -1) / 320 - 1
            cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, self.input_dim, -1) / 320 - 1
            action_net = batch[9].to(device=device, dtype=torch.float32)
            last_latent = batch[10].to(device=device, dtype=torch.float32).view(bsz, 8)

            tcp_rv = batch[11].to(device=device, dtype=torch.float32).view(bsz, 3, -1)
            tcp_theta = batch[12].to(device=device, dtype=torch.float32).unsqueeze(1)
            tcp_translation = batch[13].to(device=device, dtype=torch.float32).view(bsz, 3, -1)

            action_norm = torch.norm(action_net, dim=-1, keepdim=True)
            action_vec = action_net / (action_norm + 1e-7)

            # input = torch.cat(
            #     [cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1), action_norm.view(bsz, 1, -1)],
            #     dim=1).permute(0, 2, 1)
            input = torch.cat([cam_obs_cur, cam_obs_goal, action_vec.view(bsz, 6, -1),
                               action_norm.view(bsz, 1, -1), tcp_translation, tcp_rv, tcp_theta], dim=1).permute(0, 2,
                                                                                                                   1)

            # input = torch.cat([cam_obs_cur, action_net], dim=1).permute(0,2,1)
            output = self.adaptionmodule(input)
            # Compute actor loss
            reg_loss = self.criterion(env_latent, output)

            # ll = self.adaptionmodule.latent_loss(self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma)
            # pre_mu_std = torch.cat((self.adaptionmodule.z_mean, self.adaptionmodule.z_sigma), axis=1)
            # true_mu_std = torch.cat((self.encoder.z_mean, self.encoder.z_sigma), axis=1)
            # ml = self.adaption_criterion(pre_mu_std, true_mu_std)
            # actor_loss = ml
            # actor_loss = ll + ml
            # Optimize the actor

            traj_gt = batch[6].to(device=device, dtype=torch.float32)
            traj_cam_obs_cur = batch[7].to(device=device, dtype=torch.float32).view(bsz, 10, self.input_dim) / 320 - 1
            traj_cam_obs_goal = batch[8].to(device=device, dtype=torch.float32).view(bsz, 10, self.input_dim) / 320 - 1
            list_latent = []
            for i in range(bsz):
                list_latent.append(output[i].repeat(10,1).unsqueeze(0))
            cal_latent = torch.cat(list_latent,dim = 0)
            input = torch.cat([traj_cam_obs_cur, traj_cam_obs_goal, cal_latent], dim=2)
            vel_vec, vel_norm = self.transfermodule(input)
            # Compute actor losse
            dir_loss = scaled_cosine_similarity_loss(vel_vec, traj_gt).mean()
            gt_norm = torch.norm(traj_gt, dim=-1, keepdim=True)
            norm_loss = F.mse_loss(
                vel_norm,
                reverse_alpha_p_elu(gt_norm, eps=1e-5),
                reduction="mean"
            ) * 0.5
            actor_loss = dir_loss + norm_loss + reg_loss

            self.adaption_optimizer.zero_grad()

            actor_loss.backward()
            self.adaption_optimizer.step()
            # self.controller_optimizer.step()
            total_actor_loss += actor_loss
            total_reg_loss += reg_loss
            total_dir_loss += dir_loss
            total_norm_loss += norm_loss
            count += 1
        if not tb == None:
            tb.add_scalar('average_batch_loss', total_actor_loss / count, episode_timesteps)
            tb.add_scalar('average_reg_loss', total_reg_loss / count, episode_timesteps)
            tb.add_scalar('average_dir_loss', total_dir_loss / count, episode_timesteps)
            tb.add_scalar('average_norm_loss', total_norm_loss / count, episode_timesteps)
        return

    def save_adaption(self, filename):
        torch.save(self.adaptionmodule.state_dict(), filename)