from omni.isaac.kit import SimulationApp

# simulation_app = SimulationApp({"headless": False})

import argparse
import sys

import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix, rot_matrix_to_quat
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.types import ArticulationAction

from omni.isaac.core.prims import XFormPrim, RigidPrim
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
from omni.isaac.motion_generation.interface_config_loader import (
    get_supported_robot_policy_pairs,
    load_supported_motion_policy_config,
)
from pbvs import PBVSNumpy
import sys
sys.path.append("/home/cyx/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/exts/omni.isaac.robot_assembler")
from omni.isaac.robot_assembler import RobotAssembler
from scipy.spatial.transform import Rotation as R


parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

assets_root_path = "/home/cyx/project/visual_servo/assets"
ur5_asset_path = assets_root_path + "/ur5/ur5_gripper.usd"

peg_asset_path = assets_root_path + "/peg_and_hole/peg/peg.usd"
hole_asset_path = assets_root_path + "/peg_and_hole/hole/hole.usd"

world = World(physics_dt = 1.0 / 60.0, stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

ur5_robot_usd = add_reference_to_stage(usd_path=ur5_asset_path, prim_path="/World/ur5_robot")
add_reference_to_stage(usd_path=hole_asset_path, prim_path="/World/hole")
peg_usd = add_reference_to_stage(usd_path=peg_asset_path, prim_path="/World/peg")
ur5_robot = world.scene.add(Robot(prim_path="/World/ur5_robot", name="ur5_robot"))
hole = world.scene.add(XFormPrim(prim_path="/World/hole", name="hole", position=[0.5, 0.0, 0.0]))
peg = world.scene.add(XFormPrim(prim_path="/World/peg/peg", name="peg"))

robot_assembler = RobotAssembler()
relative_pos = np.array([0.0,0.0,0.12])
relative_ori = np.array([1.0,0.0,0.0,0.0])
relative_trans_matrix = np.eye(4)
relative_trans_matrix[:3, :3] = quat_to_rot_matrix(relative_ori)
relative_trans_matrix[:3, 3] = relative_pos
assembled_robot = robot_assembler.assemble_articulations(
        "/World/ur5_robot",
        "/World/peg",
        "/tool0",
        "/peg",
        relative_pos,
        relative_ori,
        mask_all_collisions = True,
        single_robot=False
)

world.reset()

print(hole)
ur5_robot_jnt_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"
                         , "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
gripper_jnt_names = ["finger_joint", "right_outer_knuckle_joint"]

print(ur5_robot.dof_names)

ur5_robot_jnt_idx = np.array([ur5_robot.get_dof_index(name) for name in ur5_robot_jnt_names])
gripper_jnt_idx = np.array([ur5_robot.get_dof_index(name) for name in gripper_jnt_names])

print(ur5_robot_jnt_idx)
print(gripper_jnt_idx)

ur5_robot_default_position = np.array([0, -np.pi/2, np.pi/2, 0, 0, 0])
gripper_default_position = np.array([0.523, 0.523])

ur5_robot.set_joint_positions(ur5_robot_default_position, ur5_robot_jnt_idx)
ur5_robot.set_joint_positions(gripper_default_position, gripper_jnt_idx)

world.step()

rmp_config = load_supported_motion_policy_config("UR5", "RMPflow")
rmp_config.update({"end_effector_frame_name": "tool0"})
rmpflow = RmpFlow(**rmp_config)
articulation_rmpflow = ArticulationMotionPolicy(ur5_robot, rmpflow)

end_effector = XFormPrim(prim_path="/World/ur5_robot/tool0")
target_pos, target_ori = end_effector.get_world_pose()
peg_pos, peg_ori = peg.get_world_pose()

target = XFormPrim("/World/target", position=target_pos, orientation = target_ori)

count = 0
flag = 1

# pbvs controller
desired_pos = np.array([0.5, 0.0, 0.2])
desired_ori = euler_angles_to_quat(np.array([np.pi, 0, np.pi]))
desired_trans_matrix = np.eye(4)
desired_trans_matrix[:3, :3] = quat_to_rot_matrix(desired_ori)
desired_trans_matrix[:3, 3] = desired_pos
pbvs = PBVSNumpy()
flag = 0
while simulation_app.is_running():
    
    count += 1
    # print(count)
    if count > 100:
        peg_pos, peg_ori = peg.get_world_pose()
        current_trans_matrix = np.eye(4)
        current_trans_matrix[:3, :3] = quat_to_rot_matrix(peg_ori)
        current_trans_matrix[:3, 3] = peg_pos
        
        vel = pbvs.cal_action(desired_trans_matrix, current_trans_matrix)
        # print(np.linalg.norm(vel))
        dT = np.eye(4)
        dT[:3, :3] = R.from_rotvec(vel[3:]).as_matrix()
        dT[:3, 3] = vel[:3]
        next_trans_matrix = current_trans_matrix @ dT
        next_trans_matrix_tcp = next_trans_matrix @ np.linalg.inv(relative_trans_matrix)
        target_pos = next_trans_matrix_tcp[:3, 3]
        target_ori = rot_matrix_to_quat(next_trans_matrix_tcp[:3, :3])
        target.set_world_pose(target_pos, target_ori)
    # trans_matrix = 
    # if count % 100 == 0:
    #     flag = -flag
    # ur5_robot_default_position = ur5_robot_default_position + 0.01 * flag
    target_position, target_orientation = target.get_world_pose()
    rmpflow.set_end_effector_target(
            target_position, target_orientation
        )
    rmpflow.update_world()
    robot_base_translation,robot_base_orientation = ur5_robot.get_world_pose()
    rmpflow.set_robot_base_pose(robot_base_translation,robot_base_orientation)
    action = articulation_rmpflow.get_next_articulation_action()
    # ur5_robot.apply_action(
    #     ArticulationAction(joint_positions=ur5_robot_default_position, joint_indices=ur5_robot_jnt_idx)
    # )
    
    ur5_robot.apply_action(action)
    # print("physics dt {}".format(world.get_physics_dt()))
    # print("rendering dt {}".format(world.get_rendering_dt()))
    print("simulation_time: {}".format(world.current_time))
    world.step(render=True)
    

world.reset()




