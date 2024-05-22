
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
from env.gripper_only_env import env

assets_root_path = "/home/cyx/project/visual_servo/assets"
env = env(assets_root_path, render=True, physics_dt = 1 / 60.0)
while simulation_app.is_running():
    env.step()
    observation = env.get_observation()
    if observation != None:
        env.apply_vel(observation[0])
      



            
        