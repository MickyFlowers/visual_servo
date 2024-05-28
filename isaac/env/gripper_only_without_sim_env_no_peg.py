import numpy as np
from utils.rotations import euler_angles_to_quat, calc_trans_matrix, quat_to_rot_matrix, rot_matrix_to_quat, calc_pose_from_vel
from utils.rotations import transform_points
from env.feature import feature
from utils.pbvs import PBVSNumpy

class env:
    def __init__(self) -> None:
        self.count = 0
        self.width = 640
        self.height = 480
        self.camera_to_end_effector_pos = np.array([0.05, 0.0, 0.02])
        self.camera_to_end_effector_ori = euler_angles_to_quat(np.array([0.0, 0.0, np.pi /2]), extrinsic=False)
        self.camera_to_end_effector_trans_matrix = calc_trans_matrix(self.camera_to_end_effector_pos, self.camera_to_end_effector_ori)
        
        self.hole_pos = np.array([0, 0, 0])
        self.hole_ori = euler_angles_to_quat(np.array([0, 0, 0]), extrinsic=False)
        self.hole_trans_matrix = calc_trans_matrix(self.hole_pos, self.hole_ori)

        self.camera_intrinsics = [[616.56402588,   0.        , 330.48983765],
                                [  0.        , 616.59606934, 233.84162903],
                                [  0.        ,   0.        ,   1.        ]]
        
        self.hole_feature_points = np.array([[0.045, 0.015, 0.03], 
                                        [0.015, -0.015, 0.03], 
                                        [0.045, -0.015, 0.03], 
                                        [0.015, 0.015, 0.03]])
        self.peg_feature_points = np.array([[0.015, 0.015, 0.1],
                                        [0.015, -0.015, 0.1],
                                        [0.015, 0.015, 0.07],
                                        [0.015, -0.015, 0.07]])
        self.feature = feature(self.hole_feature_points, self.peg_feature_points, self.camera_intrinsics)
        self.pbvs_controller = PBVSNumpy()

        self.reset()
    
    def reset(self):
        self.count = 0
        self.sample()
    


    def get_observation(self):
        # self.sample_desired_feature()
        self.cur_camera_trans_matrix = self.current_end_effector_trans_matrix @ self.camera_to_end_effector_trans_matrix
        self.cur_peg_trans_matrix = self.current_end_effector_trans_matrix @ self.in_hand_error_matrix
        hole_feature_points_to_world = transform_points(self.hole_feature_points, self.hole_trans_matrix)
        vel, (_, _) = self.pbvs_controller.cal_action_curve(self.cur_camera_trans_matrix, 
                                                    self.desired_camera_trans_matrix,
                                                    hole_feature_points_to_world)
        hole_feature_to_image, peg_feature_to_image = self.feature.project_points_to_img(self.cur_camera_trans_matrix, self.hole_trans_matrix, self.cur_peg_trans_matrix)          
        hole_desired_feature, peg_desired_feature = self.feature.project_points_to_img(self.ref_camera_trans_matrix, self.hole_trans_matrix, self.hole_trans_matrix @ self.desired_trans_matrix_to_hole)
        
        if np.max(hole_feature_to_image, axis=0)[0] > self.width or np.min(hole_feature_to_image, axis=0)[0] < 0:
            return False
        if np.max(hole_feature_to_image, axis=0)[1] > self.height or np.min(hole_feature_to_image, axis=0)[1] < 0:
            return False
        if np.max(hole_desired_feature, axis=0)[0] > self.width or np.min(hole_desired_feature, axis=0)[0] < 0:
            return False
        if np.max(hole_desired_feature, axis=0)[1] > self.height or np.min(hole_desired_feature, axis=0)[1] < 0:
            return False
        if np.linalg.norm(vel) < 0.003:
            return True
        if self.count > 1000:
            return False
        
        
        
        return vel, hole_desired_feature, peg_desired_feature, hole_feature_to_image, peg_feature_to_image




    def apply_vel(self, vel, dt = 1.0 / 60.0):
        self.count += 1
        next_camera_trans_matrix = calc_pose_from_vel(self.cur_camera_trans_matrix, vel, dt)
        next_gripper_trans_matrix = next_camera_trans_matrix @ np.linalg.inv(self.camera_to_end_effector_trans_matrix)
        self.current_end_effector_trans_matrix = next_gripper_trans_matrix

    def sample(self):
        # sample in hand error
        in_hand_pos_upper = np.array([0.0, 0.0, 0.10])
        in_hand_pos_lower = np.array([0.0, 0.0, 0.10])
        in_hand_rot_upper = np.array([0.0, 0.0, 0.0])
        in_hand_rot_lower = np.array([0.0, 0.0, 0.0])
        # in_hand_pos_upper = np.array([0.005, 0.0, 0.14])
        # in_hand_pos_lower = np.array([-0.005, 0.0, 0.10])
        # in_hand_rot_upper = np.array([0.0, 0.3, 0.0])
        # in_hand_rot_lower = np.array([0.0, -0.1, 0.0])

        self.in_hand_error_pos = np.random.uniform(in_hand_pos_lower, in_hand_pos_upper)
        self.in_hand_error_ori = np.random.uniform(in_hand_rot_lower, in_hand_rot_upper)
        self.in_hand_error_ori = euler_angles_to_quat(self.in_hand_error_ori, extrinsic=False)
        self.in_hand_error_matrix = calc_trans_matrix(self.in_hand_error_pos, self.in_hand_error_ori)

        self.target_in_hand_error_pos = np.random.uniform(in_hand_pos_lower, in_hand_pos_upper)
        self.target_in_hand_error_ori = np.random.uniform(in_hand_rot_lower, in_hand_rot_upper)
        self.target_in_hand_error_ori = euler_angles_to_quat(self.target_in_hand_error_ori, extrinsic=False)
        self.target_in_hand_error_matrix = calc_trans_matrix(self.target_in_hand_error_pos, self.target_in_hand_error_ori)

        # sample init pose

        default_ori = euler_angles_to_quat(np.array([np.pi, 0.0, np.pi/2]), extrinsic=False)
        default_rot_matrix = quat_to_rot_matrix(default_ori)
        default_pos = np.array([0.03, 0.0, 0.03])
        

        init_relative_rot_lower = np.array([-np.pi/6, -np.pi/6, -np.pi/6])
        init_relative_rot_upper = np.array([np.pi/6, np.pi/6, np.pi/6])
        disturb_rot_lower = np.array([-np.pi/6, -np.pi/6, 0])
        disturb_rot_upper = np.array([np.pi/6, np.pi/6, 0])


        init_relative_rot = np.random.uniform(init_relative_rot_lower, init_relative_rot_upper)
        init_relative_ori = euler_angles_to_quat(init_relative_rot, extrinsic=False)
        init_relative_rot_matrix = quat_to_rot_matrix(init_relative_ori)  

        disturb_rot = np.random.uniform(disturb_rot_lower, disturb_rot_upper)
        disturb_ori = euler_angles_to_quat(disturb_rot, extrinsic=False)
        disturb_rot_matrix = quat_to_rot_matrix(disturb_ori)

        init_pose_rot_matrix = default_rot_matrix @ init_relative_rot_matrix @ disturb_rot_matrix

        r = np.random.uniform(0.4, 0.5)
        dx= r * np.sin(-init_relative_rot[0])
        dy = r * np.abs(np.cos(init_relative_rot[0])) * np.sin(init_relative_rot[1])
        dz = np.sqrt(r**2 - dx**2 - dy**2)
        init_pose_pos = default_pos + np.array([dx, dy, dz])

        init_camera_trans_matrix = np.eye(4)
        init_camera_trans_matrix[:3, :3] = init_pose_rot_matrix
        init_camera_trans_matrix[:3, 3] = init_pose_pos

        self.init_end_effector_trans_matrix = init_camera_trans_matrix @ np.linalg.inv(self.camera_to_end_effector_trans_matrix)
        self.init_end_effector_pos = self.init_end_effector_trans_matrix[:3, 3]
        self.init_end_effector_ori = rot_matrix_to_quat(self.init_end_effector_trans_matrix[:3, :3])

        self.current_end_effector_trans_matrix = self.init_end_effector_trans_matrix
        self.current_end_effector_pos = self.init_end_effector_pos
        self.current_end_effector_ori = self.init_end_effector_ori
        self.sample_desired_feature()

    def sample_desired_feature(self):
        # sample relative pose
        desired_pos_upper = np.array([0.01, 0.01, 0.1])
        desired_pos_lower = np.array([-0.01, -0.01, 0.05])
        desired_rot_upper = np.array([np.pi/6, np.pi/6, np.pi/6])
        desired_rot_lower = np.array([-np.pi/6, 0, -np.pi/6])

        self.desired_pos_to_hole = np.random.uniform(desired_pos_lower, desired_pos_upper)
        self.desired_pos_to_hole = self.desired_pos_to_hole + np.array([0, 0, 0.2])
        self.desired_ori_to_hole = np.random.uniform(desired_rot_lower, desired_rot_upper)
        # self.desired_ori_to_hole = np.array([0, 0, 0])
        self.desired_ori_to_hole = euler_angles_to_quat(self.desired_ori_to_hole, extrinsic=False)
        self.desired_rot_matrix_to_hole = quat_to_rot_matrix(self.desired_ori_to_hole)
        self.desired_rot_matrix_to_hole = quat_to_rot_matrix(euler_angles_to_quat(np.array([np.pi, 0, 0 ]), extrinsic=False)) @ self.desired_rot_matrix_to_hole 
        self.desired_ori_to_hole = rot_matrix_to_quat(self.desired_rot_matrix_to_hole)
        self.desired_trans_matrix_to_hole = calc_trans_matrix(self.desired_pos_to_hole, self.desired_ori_to_hole)
        self.desired_camera_trans_matrix = self.hole_trans_matrix @ self.desired_trans_matrix_to_hole @ np.linalg.inv(self.in_hand_error_matrix) @ self.camera_to_end_effector_trans_matrix
        self.ref_camera_trans_matrix = self.hole_trans_matrix @ self.desired_trans_matrix_to_hole @ np.linalg.inv(self.target_in_hand_error_matrix) @ self.camera_to_end_effector_trans_matrix
