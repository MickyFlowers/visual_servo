from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})
from omni.isaac.core.utils.rotations import euler_angles_to_quat, rot_matrix_to_quat, quat_to_rot_matrix, quat_to_euler_angles
from env.feature import feature
import numpy as np
import os
print(os.getcwd())
from utils.rotations import calc_trans_matrix
import matplotlib.pyplot as plt
data = []
if __name__ == "__main__":
    # define the hole and peg points
    count = 0
    
    while count < 1e6:
        dict = {}
        hole_feature_points = np.array([[0.045, 0.015, 0.03], 
                                        [0.015, -0.015, 0.03], 
                                        [0.045, -0.015, 0.03], 
                                        [0.015, 0.015, 0.03]])
        peg_feature_points = np.array([[0.015, 0.015, 0.1],
                                    [0.015, -0.015, 0.1],
                                    [0.015, 0.015, 0.07],
                                    [0.015, -0.015, 0.07]])
        camera_intrinsics = [[616.56402588,   0.        , 330.48983765],
                            [  0.        , 616.59606934, 233.84162903],
                            [  0.        ,   0.        ,   1.        ]]
        # create a feature object
        feature_obj = feature(hole_feature_points, peg_feature_points, camera_intrinsics)

        in_hand_pos_upper = np.array([0.005, 0.0, 0.14])
        in_hand_pos_lower = np.array([-0.005, 0.0, 0.10])
        in_hand_rot_upper = np.array([0.0, 0.3, 0.0])
        in_hand_rot_lower = np.array([0.0, -0.1, 0.0])

        in_hand_error_pos = np.random.uniform(in_hand_pos_lower, in_hand_pos_upper)
        in_hand_error_ori = np.random.uniform(in_hand_rot_lower, in_hand_rot_upper)
        in_hand_error_ori = euler_angles_to_quat(in_hand_error_ori, extrinsic=False)
        in_hand_error_matrix = calc_trans_matrix(in_hand_error_pos, in_hand_error_ori)
        
        desired_pos_upper = np.array([0.01, 0.01, 0.1])
        desired_pos_lower = np.array([-0.01, -0.01, 0.05])
        desired_rot_upper = np.array([np.pi/12, np.pi/12, np.pi/12])
        desired_rot_lower = np.array([-np.pi/12, 0, -np.pi/12])

        desired_pos_to_hole = np.random.uniform(desired_pos_lower, desired_pos_upper)
        desired_pos_to_hole = desired_pos_to_hole + np.array([0, 0, 0.2])
        desired_ori_to_hole = np.random.uniform(desired_rot_lower, desired_rot_upper)
        desired_ori_to_hole = euler_angles_to_quat(desired_ori_to_hole, extrinsic=False)
        desired_rot_matrix_to_hole = quat_to_rot_matrix(desired_ori_to_hole)
        desired_rot_matrix_to_hole = quat_to_rot_matrix(euler_angles_to_quat(np.array([np.pi, 0, 0 ]), extrinsic=False)) @ desired_rot_matrix_to_hole 
        desired_ori_to_hole = rot_matrix_to_quat(desired_rot_matrix_to_hole)
        desired_trans_matrix_to_hole = calc_trans_matrix(desired_pos_to_hole, desired_ori_to_hole)

        camera_to_end_effector_pos = np.array([0.05, 0.0, 0.02])
        camera_to_end_effector_ori = euler_angles_to_quat(np.array([0.0, 0.0, np.pi /2]), extrinsic=False)
        camera_to_end_effector_trans_matrix = calc_trans_matrix(camera_to_end_effector_pos, camera_to_end_effector_ori)

        hole_pos = np.array([0, 0, 0])
        hole_ori = euler_angles_to_quat(np.array([0, 0, 0]), extrinsic=False)
        hole_trans_matrix = calc_trans_matrix(hole_pos, hole_ori)

        peg_trans_matrix = hole_trans_matrix @ desired_trans_matrix_to_hole
        end_effector_trans_matrix = peg_trans_matrix @ np.linalg.inv(in_hand_error_matrix)
        camera_trans_matrix = end_effector_trans_matrix @ camera_to_end_effector_trans_matrix

        hole_feature_points_to_image, peg_feature_points_to_image = feature_obj.project_points_to_img(camera_trans_matrix, hole_trans_matrix, peg_trans_matrix)
        dict["hole_feature_points_to_image"] = hole_feature_points_to_image
        dict["peg_feature_points_to_image"] = peg_feature_points_to_image
        if np.max(hole_feature_points_to_image, axis=0)[0] > 640 or np.min(hole_feature_points_to_image, axis=0)[0] < 0:
            continue
        if np.max(hole_feature_points_to_image, axis=0)[1] > 480 or np.min(hole_feature_points_to_image, axis=0)[1] < 0:
            continue
        # print(hole_feature_points_to_image)
        # figure = plt.figure()
        # plt.plot(hole_feature_points_to_image[:, 0], hole_feature_points_to_image[:, 1], 'ro')
        # plt.show()
        data.append(dict)
        count += 1
    # save data
    import datetime
    np.save("./data/desired_feature" + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + ".npy", data)
