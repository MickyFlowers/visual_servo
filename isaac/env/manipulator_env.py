import sys
sys.path.append("/home/cyx/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/exts/omni.isaac.robot_assembler")
import numpy as np
import ikfastpy.ikfastpy as ikfastpy
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
from omni.isaac.motion_generation.interface_config_loader import (
    get_supported_robot_policy_pairs,
    load_supported_motion_policy_config,
)
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
from utils.rotations import calc_trans_matrix, transform_points


class env:
    def __init__(self, assets_root_path, render, physics_dt = 1 / 60.0):
        self.count = 0
        self.render = render
        self.world = World(physics_dt = physics_dt, stage_units_in_meters = 1.0)
        self.world.reset()
        self.world.scene.add_default_ground_plane()
        ur5_asset_path = assets_root_path + "/ur5/ur5_gripper.usd"
        peg_asset_path = assets_root_path + "/peg_and_hole/peg/peg.usd"
        hole_asset_path = assets_root_path + "/peg_and_hole/hole/hole.usd"

        self.camera_pos = np.array([0.05, 0.0, 0.02])
        self.camera_ori = euler_angles_to_quat(np.array([0.0, 5.0/180.0 * np.pi, np.pi /2]), extrinsic=False)
        self.camera_matrix = calc_trans_matrix(self.camera_pos, self.camera_ori)

        self.ur5_robot_usd_prim = add_reference_to_stage(usd_path = ur5_asset_path, prim_path="/World/ur5_robot")
        self.peg_usd_prim = add_reference_to_stage(usd_path = peg_asset_path, prim_path="/World/peg")
        self.hole_usd_prim = add_reference_to_stage(usd_path = hole_asset_path, prim_path="/World/hole")
        
        self.ur5_robot = self.world.scene.add(Robot(prim_path="/World/ur5_robot", name="ur5_robot"))
        self.hole_set = self.world.scene.add(XFormPrim(prim_path="/World/hole", name="hole", position=[0.5, 0.0, 0.0]))
        self.peg = self.world.scene.add(XFormPrim(prim_path="/World/peg/peg", name="peg"))

        
        self.hole = XFormPrim(prim_path="/World/hole/hole")

        self.robot_assembler = RobotAssembler()

        self.in_hand_pos_upper = np.array([0.005, 0.0, 0.14])
        self.in_hand_pos_lower = np.array([-0.005, 0.0, 0.10])
        self.in_hand_rot_upper = np.array([0.0, 0.3, 0.0])
        self.in_hand_rot_lower = np.array([0.0, -0.1, 0.0])

        self.points_in_hole = np.array([[0.0155, 0.015, 0.03], 
                                        [0.0155, -0.015, 0.03], 
                                        [0.0455, 0.015, 0.03], 
                                        [0.0455, -0.015, 0.03]])
        self.points_in_peg = np.array([[0.015, 0.015, 0.1],
                                        [0.015, -0.015, 0.1],
                                        [0.015, 0.015, 0.07],
                                        [0.015, -0.015, 0.07]])
        
        self.in_hand_pos, self.in_hand_ori = self.sample_in_hand_pose(self.in_hand_pos_upper, self.in_hand_pos_lower, self.in_hand_rot_upper, self.in_hand_rot_lower)
        self.in_hand_trans_matrix = calc_trans_matrix(self.in_hand_pos, self.in_hand_ori)
        
        self.assembled_robot = self.robot_assembler.assemble_articulations(
            "/World/ur5_robot",
            "/World/peg",
            "/tool0",
            "/peg",
            self.in_hand_pos,
            self.in_hand_ori,
            mask_all_collisions = True,
            single_robot=False
        )

        #config kinematics
        from omni.isaac.core.utils.extensions import get_extension_path_from_name
        
        mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
        

        self.ur5_robot_jnt_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"
                         , "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.gripper_jnt_names = ["finger_joint", "right_outer_knuckle_joint"]

        
        self.ur5_robot_default_position = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2])
        self.gripper_default_position = np.array([0.523, 0.523])

        rmp_config = load_supported_motion_policy_config("UR5", "RMPflow")
        
        rmp_config.update({"end_effector_frame_name": "tool0"})
        self.rmpflow = RmpFlow(**rmp_config)
        # self.rmpflow.add_obstacle(self.hole)
        self.articulation_rmpflow = ArticulationMotionPolicy(self.ur5_robot, self.rmpflow)
        self.end_effector = RigidPrim(prim_path="/World/ur5_robot/tool0")
        self.camera = Camera(prim_path="/World/ur5_robot/tool0/Camera", resolution=(640, 480))
        # self.reset()
        self.ur5_ik_solver = ikfastpy.PyKinematics()
        self.world.reset()
        self.reset()
        
    def sample_in_hand_pose(self, pos_upper, pos_lower, rot_upper, rot_lower):
        pos = np.random.uniform(pos_upper, pos_lower)
        rot = np.random.uniform(rot_upper, rot_lower)
        ori = euler_angles_to_quat(rot, extrinsic=False)
        return pos, ori
    
    def set_camera_parameter(self):
        width, height = 1920, 1200
        camera_matrix = [[616.56402588,   0.        , 330.48983765],
                        [  0.        , 616.59606934, 233.84162903],
                        [  0.        ,   0.        ,   1.        ]]
        self.feature = feature(self.points_in_hole, self.points_in_hole, camera_matrix)
        # Pixel size in microns, aperture and focus distance from the camera sensor specification
        # Note: to disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
        pixel_size = 3 * 1e-3   # in mm, 3 microns is a common pixel size for high resolution cameras
        f_stop = 1.8            # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
        focus_distance = 0.6    # in meters, the distance from the camera to the object plane

        # Calculate the focal length and aperture size from the camera matrix
        ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
        horizontal_aperture =  pixel_size * width                   # The aperture size in mm
        vertical_aperture =  pixel_size * height
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
        
    def reset(self):

        self.reset_first = 0
        self.count = 0
        self.camera.initialize()
        self.set_camera_parameter()
        self.in_hand_pos, self.in_hand_ori = self.sample_in_hand_pose(self.in_hand_pos_upper, self.in_hand_pos_lower, self.in_hand_rot_upper, self.in_hand_rot_lower)
        self.in_hand_trans_matrix = calc_trans_matrix(self.in_hand_pos, self.in_hand_ori)
        self.assembled_robot.set_fixed_joint_transform(self.in_hand_pos,self.in_hand_ori)
        self.world.reset()
        
        self.desired_pos_to_hole, self.desired_rot_to_hole = self.sample_desired_pose()

        self.ur5_robot_jnt_idx = np.array([self.ur5_robot.get_dof_index(name) for name in self.ur5_robot_jnt_names])
        self.gripper_jnt_idx = np.array([self.ur5_robot.get_dof_index(name) for name in self.gripper_jnt_names])
        # print(self.ur5_robot_jnt_idx)
        joint_positions = np.array(self.sample_ur5_init_pose())
        # self.ur5_robot.set_joint_positions(self.ur5_robot_default_position, self.ur5_robot_jnt_idx)
        self.ur5_robot.set_joint_positions(joint_positions, self.ur5_robot_jnt_idx)
        
        self.ur5_robot.set_joint_positions(self.gripper_default_position, self.gripper_jnt_idx)
        
        # self.world.step(render=self.render)
        self.target_pos, self.target_ori = self.end_effector.get_world_pose()

        self.target = XFormPrim("/World/target", position=self.target_pos, orientation = self.target_ori)


    def sample_desired_pose(self):
        pos_upper = np.array([0.05, 0.05, 0.1])
        pos_lower = np.array([-0.05, -0.05, 0.05])
        rot_upper = np.array([np.pi/6, np.pi/6, np.pi/6])
        rot_lower = np.array([-np.pi/6, -np.pi/6, -np.pi/6])
        desired_pos_to_hole = np.random.uniform(pos_upper, pos_lower)
        desired_rot_to_hole = euler_angles_to_quat(np.random.uniform(rot_upper, rot_lower), extrinsic=False)
        return desired_pos_to_hole, desired_rot_to_hole
        

    def step(self):
        self.count += 1
        self.world.step(render=self.render)
        if self.count > 20:
            
            # print(self.ur5_robot.get_joint_positions(self.ur5_robot_jnt_idx))
            #project to img
            hole_pos, hole_ori = self.hole.get_world_pose()
            hole_pose_trans_matrix = calc_trans_matrix(hole_pos, hole_ori)
            hole_world_points = transform_points(self.points_in_hole, hole_pose_trans_matrix)

            end_effector_pos, end_effector_ori = self.end_effector.get_world_pose()
            end_effector_matrix = calc_trans_matrix(end_effector_pos, end_effector_ori)
            camera_pose_matrix = end_effector_matrix @ self.camera_matrix
            camera_pos = camera_pose_matrix[:3, 3]
            camera_ori = rot_matrix_to_quat(camera_pose_matrix[:3, :3])
            peg_pos, peg_ori = self.peg.get_world_pose()
            peg_pose_trans_matrix = calc_trans_matrix(peg_pos, peg_ori)
            peg_world_points = transform_points(self.points_in_peg, peg_pose_trans_matrix)
            # print(np.linalg.inv(peg_pose_trans_matrix) @ camera_pose_matrix)
            # hole_img_points, peg_img_points = self.feature.project_points_to_img(camera_pos, camera_ori, hole_pos, hole_ori, peg_pos, peg_ori)
            
            # peg_img_points = self.camera.get_image_coords_from_world_points(peg_world_points)
            # hole_img_points = self.camera.get_image_coords_from_world_points(np.array([[0.5, 0, 0]]))
            # print(peg_img_points)
            # print(hole_img_points)
            #desired pose
            desired_pos = hole_pos + self.desired_pos_to_hole + np.array([0, 0, 0.2])
            desired_ori_matrix = quat_to_rot_matrix(hole_ori) @ quat_to_rot_matrix(euler_angles_to_quat(np.array([np.pi, 0, 0 ]), extrinsic=False)) @ quat_to_rot_matrix(self.desired_rot_to_hole)
            desired_ori = rot_matrix_to_quat(desired_ori_matrix)
            desired_matrix = calc_trans_matrix(desired_pos, desired_ori)
            end_effector_target_matrix = desired_matrix @ np.linalg.inv(self.in_hand_trans_matrix)
            end_effector_target_pos = end_effector_target_matrix[:3, 3]
            end_effector_target_ori = rot_matrix_to_quat(end_effector_target_matrix[:3, :3])
            camera_target_matrix = end_effector_target_matrix @ self.camera_matrix
            camera_target_pos = camera_target_matrix[:3, 3]
            camera_target_ori = rot_matrix_to_quat(camera_target_matrix[:3, :3])
            # if self.reset_first == 0:
            #     hole_img_target_points, peg_img_target_points = self.feature.project_points_to_img(camera_target_pos, camera_target_ori, hole_pos, hole_ori, desired_pos, desired_ori)
            #     self.reset_first = 1
            # print("peg pos: ", peg_pos)
            self.target.set_world_pose(self.target_pos, self.target_ori)
            self.rmpflow.set_end_effector_target(end_effector_target_pos, end_effector_target_ori)
            self.rmpflow.update_world()
            robot_base_translation,robot_base_orientation = self.ur5_robot.get_world_pose()
            self.rmpflow.set_robot_base_pose(robot_base_translation,robot_base_orientation)
            action = self.articulation_rmpflow.get_next_articulation_action(self.world.get_physics_dt())
            self.ur5_robot.apply_action(action)
            # print(self.ur5_robot.get_joint_positions(self.ur5_robot_jnt_idx))
            # print(self.end_effector.get_world_pose())
            # print("pos_error", np.linalg.norm(peg_pos - desired_pos))
            # print("rot_error", np.linalg.norm(peg_ori - desired_ori))
            if np.linalg.norm(peg_pos - desired_pos) < 0.001 and np.linalg.norm(peg_ori - desired_ori) < 0.003:
                self.reset()
            


    def sample_ur5_init_pose(self):
        #sample camera pose in a sphere
        rot_base = np.array([np.pi, 0.0, np.pi/2])
        pos_base = np.array([0.575, 0.0, 0.03])
        rot_lower = np.array([-np.pi/6, -np.pi/6, -np.pi/4])
        rot_upper = np.array([np.pi/6, np.pi/6, np.pi/4])

        disturb_rot_lower = np.array([-np.pi/12, -np.pi/12, 0])
        disturb_rot_upper = np.array([np.pi/12, np.pi/12, 0])
        disturb_rot = np.random.uniform(disturb_rot_lower, disturb_rot_upper)
        rot = np.random.uniform(rot_upper, rot_lower)
        
        # rot = np.array([0.0, np.pi / 6, 0.0])
        r = np.random.uniform(0.4, 0.5)
        dx= r * np.sin(-rot[0])
        dy = r * np.abs(np.cos(rot[0])) * np.sin(rot[1])
        dz = np.sqrt(r**2 - dx**2 - dy**2)
        pos = pos_base + np.array([dx, dy, dz])
        
        rot_base_matrix = quat_to_rot_matrix(euler_angles_to_quat(rot_base, extrinsic=False))
        rot_relative_matrix = quat_to_rot_matrix(euler_angles_to_quat(rot, extrinsic=False))
        disturb_rot_matrix = quat_to_rot_matrix(euler_angles_to_quat(disturb_rot, extrinsic=False))
        rot_matrix = rot_base_matrix @ rot_relative_matrix @ disturb_rot_matrix
        # print(quat_to_euler_angles(rot_matrix_to_quat(rot_matrix)))
        camera_target_frame = np.eye(4)
        camera_target_frame[:3, :3] = rot_matrix
        camera_target_frame[:3, 3] = pos
        target_frame = camera_target_frame @ np.linalg.inv(self.camera_matrix)
        target_frame = calc_trans_matrix(np.array([0, 0, 0]), euler_angles_to_quat(np.array([0, 0, np.pi]) ,extrinsic=False)) @ target_frame
        joint_configs = self.ur5_ik_solver.inverse(target_frame[:3, :].reshape(-1).tolist())
        n_solutions = int(len(joint_configs) / 6)
        if n_solutions  == 0:
            return self.sample_ur5_init_pose()
        joint_configs = np.array(joint_configs).reshape(n_solutions, 6)
        joint_config_error = joint_configs - self.ur5_robot_default_position
        idx = np.argmin(np.max(np.abs(joint_config_error), axis=1))
        if np.max(np.abs(joint_config_error[idx])) > np.pi / 2:
            return self.sample_ur5_init_pose()
        return joint_configs[idx]
        # print(quat_to_euler_angles(rot_matrix_to_quat(self.camera_matrix), extrinsic=False))
        # print(quat_to_euler_angles(rot_matrix_to_quat(np.linalg.inv(self.camera_matrix)), extrinsic=False))
        # print(quat_to_euler_angles(rot_matrix_to_quat(target_frame[:3, :3]), extrinsic=False))
        # print(target_frame)

        

        
        
    



        
        

        

        
    

        
