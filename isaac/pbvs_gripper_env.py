
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})
from isaac.env.gripper_only_env_no_peg import env

assets_root_path = "/home/cyx/project/visual_servo/assets"
env = env(assets_root_path, render=False, physics_dt = 1 / 60.0)
while simulation_app.is_running():
    env.step()
    print(env.count)
    observation = env.get_observation()
    if observation != None:
        env.apply_vel(observation[0])
      



            
        