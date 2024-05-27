import sys
sys.path.append("/home/cyx/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/exts/omni.isaac.robot_assembler")
import numpy as np
import ikfastpy.ikfastpy as ikfastpy
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims import XFormPrim, RigidPrim

from omni.isaac.robot_assembler import RobotAssembler
from omni.isaac.core.utils.rotations import euler_angles_to_quat, rot_matrix_to_quat, quat_to_rot_matrix, quat_to_euler_angles
from omni.isaac.core import World
from omni.isaac.sensor import Camera
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from env.feature import feature
import os
from utils.rotations import calc_trans_matrix, transform_points, calc_pose_from_vel
from utils.pbvs import PBVSNumpy
class env:
    def __init__(self, assets_root_path, render, physics_dt = 1 / 60.0) -> None:
        self.render = render
        
        self.world = World(physics_dt = physics_dt, stage_units_in_meters = 1.0)
        self.world.scene.add_default_ground_plane()
        gripper_asset_path = assets_root_path + "/robotiq/2f85_instanceable.usd"
        peg_asset_path = assets_root_path + "/peg_and_hole/peg/peg.usd"
        hole_asset_path = assets_root_path + "/peg_and_hole/hole/hole.usd"

        self.camera_to_end_effector_pos = np.array([0.05, 0.0, 0.02])
        self.camera_to_end_effector_ori = euler_angles_to_quat(np.array([0.0, 0.0, np.pi /2]), extrinsic=False)
        self.camera_to_end_effector_trans_matrix = calc_trans_matrix(self.camera_to_end_effector_pos, self.camera_to_end_effector_ori)

        self.gripper_usd_prim = add_reference_to_stage(usd_path = gripper_asset_path, prim_path="/World/gripper")
        self.peg_usd_prim = add_reference_to_stage(usd_path = peg_asset_path, prim_path="/World/peg")
        self.hole_usd_prim = add_reference_to_stage(usd_path = hole_asset_path, prim_path="/World/hole")

        self.gripper = self.world.scene.add(Robot(prim_path="/World/gripper", name="gripper", position=[0.0, 0.0, 0.5]))
        self.hole_set = self.world.scene.add(XFormPrim(prim_path="/World/hole", name="hole", position=[0.0, 0.0, 0.0]))
        self.hole = XFormPrim(prim_path="/World/hole/hole")
        self.peg = self.world.scene.add(XFormPrim(prim_path="/World/peg/peg", name="peg"))
        # set camera
        self.width, self.height = 640, 480
        self.camera = Camera(prim_path="/World/gripper/robotiq_arg2f_base_link/Camera", resolution=(self.width, self.height))
        self.camera_intrinsics = [[616.56402588,   0.        , 330.48983765],
                        [  0.        , 616.59606934, 233.84162903],
                        [  0.        ,   0.        ,   1.        ]]
        
        # assemble peg
        robot_assembler = RobotAssembler()
        self.assembled_robot = robot_assembler.assemble_articulations(
            "/World/gripper",
            "/World/peg",
            "/robotiq_arg2f_base_link",
            "/peg",
            mask_all_collisions = True,
            single_robot=False
        )

        # add controller
        self.pbvs_controller = PBVSNumpy()
        self.reset()

        # set default points
        self.hole_feature_points = np.array([[0.045, 0.015, 0.03], 
                                        [0.015, -0.015, 0.03], 
                                        [0.045, -0.015, 0.03], 
                                        [0.015, 0.015, 0.03]])
        self.peg_feature_points = np.array([[0.015, 0.015, 0.1],
                                        [0.015, -0.015, 0.1],
                                        [0.015, 0.015, 0.07],
                                        [0.015, -0.015, 0.07]])
        self.feature = feature(self.hole_feature_points, self.peg_feature_points, self.camera_intrinsics)
        self.epoch = 0

    def reset(self):
        self.dict = {}
        self.vel_save = []
        self.cur_hole_feature_save = []
        self.cur_peg_feature_save = []
        self.count = 0
        self.sample()
        self.assembled_robot.set_fixed_joint_transform(self.in_hand_error_pos,self.in_hand_error_ori)
        self.camera.initialize()
        self.world.reset()
        # set gripper
        gripper_default_position = np.array([0.523, 0.523])
        gripper_jnt_names = ["finger_joint", "right_outer_knuckle_joint"]
        self.gripper_jnt_idx = np.array([0, 1])
        self.gripper.set_joint_positions(gripper_default_position, self.gripper_jnt_idx)
        self.gripper.set_world_pose(self.init_end_effector_pos, self.init_end_effector_ori)
        # set camera
        # Pixel size in microns, aperture and focus distance from the camera sensor specification
        # Note: to disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
        pixel_size = 3 * 1e-3   # in mm, 3 microns is a common pixel size for high resolution cameras
        f_stop = 1.8            # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
        focus_distance = 0.6    # in meters, the distance from the camera to the object plane

        # Calculate the focal length and aperture size from the camera matrix
        ((fx,_,cx),(_,fy,cy),(_,_,_)) = self.camera_intrinsics
        horizontal_aperture =  pixel_size * self.width                   # The aperture size in mm
        vertical_aperture =  pixel_size * self.height
        focal_length_x  = fx * pixel_size
        focal_length_y  = fy * pixel_size
        focal_length = (focal_length_x + focal_length_y) / 2         # The focal length in mm

        # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
        self.camera.set_focal_length(focal_length / 10.0)                # Convert from mm to cm (or 1/10th of a world unit)
        self.camera.set_focus_distance(focus_distance)                   # The focus distance in meters
        self.camera.set_lens_aperture(f_stop * 100.0)                    # Convert the f-stop to Isaac Sim units
        self.camera.set_horizontal_aperture(horizontal_aperture / 10.0)  # Convert from mm to cm (or 1/10th of a world unit)
        self.camera.set_vertical_aperture(vertical_aperture / 10.0)

        self.camera.set_clipping_range(0.05, 1.0e5)
        

    def sample(self):
        # sample in hand error
        in_hand_pos_upper = np.array([0.0, 0.0, 0.10])
        in_hand_pos_lower = np.array([-0.0, 0.0, 0.10])
        in_hand_rot_upper = np.array([0.0, 0.0, 0.0])
        in_hand_rot_lower = np.array([0.0, -0.0, 0.0])
        # in_hand_pos_upper = np.array([0.005, 0.0, 0.14])
        # in_hand_pos_lower = np.array([-0.005, 0.0, 0.10])
        # in_hand_rot_upper = np.array([0.0, 0.3, 0.0])
        # in_hand_rot_lower = np.array([0.0, -0.1, 0.0])

        self.in_hand_error_pos = np.random.uniform(in_hand_pos_lower, in_hand_pos_upper)
        self.in_hand_error_ori = np.random.uniform(in_hand_rot_lower, in_hand_rot_upper)
        self.in_hand_error_ori = euler_angles_to_quat(self.in_hand_error_ori, extrinsic=False)
        self.in_hand_error_matrix = calc_trans_matrix(self.in_hand_error_pos, self.in_hand_error_ori)

        # sample init pose

        default_ori = euler_angles_to_quat(np.array([np.pi, 0.0, np.pi/2]), extrinsic=False)
        default_rot_matrix = quat_to_rot_matrix(default_ori)
        default_pos = np.array([0.03, 0.0, 0.03])
        

        init_relative_rot_lower = np.array([-np.pi/6, -np.pi/6, -np.pi/4])
        init_relative_rot_upper = np.array([np.pi/6, np.pi/6, np.pi/4])
        disturb_rot_lower = np.array([-np.pi/12, -np.pi/12, 0])
        disturb_rot_upper = np.array([np.pi/12, np.pi/12, 0])


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

        # sample relative pose
        desired_pos_upper = np.array([0.01, 0.01, 0.1])
        desired_pos_lower = np.array([-0.01, -0.01, 0.05])
        desired_rot_upper = np.array([np.pi/12, np.pi/12, np.pi/12])
        desired_rot_lower = np.array([-np.pi/12, 0, -np.pi/12])

        self.desired_pos_to_hole = np.random.uniform(desired_pos_lower, desired_pos_upper)
        self.desired_pos_to_hole = self.desired_pos_to_hole + np.array([0, 0, 0.2])
        self.desired_ori_to_hole = np.random.uniform(desired_rot_lower, desired_rot_upper)
        # self.desired_ori_to_hole = np.array([0, 0, 0])
        self.desired_ori_to_hole = euler_angles_to_quat(self.desired_ori_to_hole, extrinsic=False)
        self.desired_rot_matrix_to_hole = quat_to_rot_matrix(self.desired_ori_to_hole)
        self.desired_rot_matrix_to_hole = quat_to_rot_matrix(euler_angles_to_quat(np.array([np.pi, 0, 0 ]), extrinsic=False)) @ self.desired_rot_matrix_to_hole 
        self.desired_ori_to_hole = rot_matrix_to_quat(self.desired_rot_matrix_to_hole)
        self.desired_trans_matrix_to_hole = calc_trans_matrix(self.desired_pos_to_hole, self.desired_ori_to_hole)
        
    def get_observation(self):
        if self.count > 10:
            cur_peg_pos, cur_peg_ori = self.peg.get_world_pose()
            cur_hole_pos, cur_hole_ori = self.hole.get_world_pose()
            cur_peg_trans_matrix = calc_trans_matrix(cur_peg_pos, cur_peg_ori)
            cur_hole_trans_matrix = calc_trans_matrix(cur_hole_pos, cur_hole_ori)
            self.cur_camera_trans_matrix = cur_peg_trans_matrix @ np.linalg.inv(self.in_hand_error_matrix) @ self.camera_to_end_effector_trans_matrix
            desired_camera_trans_matrix = cur_hole_trans_matrix @ self.desired_trans_matrix_to_hole @ np.linalg.inv(self.in_hand_error_matrix) @ self.camera_to_end_effector_trans_matrix
            hole_feature_points_to_world = transform_points(self.hole_feature_points, cur_hole_trans_matrix)
            vel, (_, _) = self.pbvs_controller.cal_action_curve(self.cur_camera_trans_matrix, 
                                                        desired_camera_trans_matrix,
                                                        hole_feature_points_to_world)
            hole_feature_to_image, peg_feature_to_image = self.feature.project_points_to_img(self.cur_camera_trans_matrix, cur_hole_trans_matrix, cur_peg_trans_matrix)          
            hole_desired_feature, peg_desired_feature = self.feature.project_points_to_img(desired_camera_trans_matrix, cur_hole_trans_matrix, cur_hole_trans_matrix @ self.desired_trans_matrix_to_hole)
            cur_camera_pos = self.cur_camera_trans_matrix[:3, 3]
            cur_camera_ori = rot_matrix_to_quat(self.cur_camera_trans_matrix[:3, :3])
            print(cur_camera_pos, cur_camera_ori)
            if np.max(hole_feature_to_image, axis=0)[0] > self.width or np.min(hole_feature_to_image, axis=0)[0] < 0:
                self.reset()
                return None
            if np.max(hole_feature_to_image, axis=0)[1] > self.height or np.min(hole_feature_to_image, axis=0)[1] < 0:
                self.reset()
                return None
            if np.linalg.norm(vel) < 0.002:
                self.reset()
                return None
            if self.count > 400:
                self.reset()
                return None
            return vel, hole_desired_feature, peg_desired_feature, hole_feature_to_image, peg_feature_to_image
        else:
            return None

    def apply_vel(self, vel):
        if self.count > 10:
            next_camera_trans_matrix = calc_pose_from_vel(self.cur_camera_trans_matrix, vel, self.world.get_physics_dt())
            next_gripper_trans_matrix = next_camera_trans_matrix @ np.linalg.inv(self.camera_to_end_effector_trans_matrix)
            next_gripper_pos = next_gripper_trans_matrix[:3, 3]
            next_gripper_ori = rot_matrix_to_quat(next_gripper_trans_matrix[:3, :3])
            self.gripper.set_world_pose(next_gripper_pos, next_gripper_ori)
            

    def step(self):
        self.world.step(render = self.render)    
        self.count += 1
        
            
        

