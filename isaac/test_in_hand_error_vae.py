test = True
from omni.isaac.kit import SimulationApp
if test == True:
    simulation_app = SimulationApp({"headless": False})
else:
    simulation_app = SimulationApp({"headless": True})
from env.gripper_only_env_peg import env
from control_policy.control_policy import control_policy
import torch
import numpy as np
from vae.vae import vae
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
if test == True:
    model = control_policy(28).to(device)
    model.load_state_dict(torch.load("./isaac/model/control_policy.pth"))
else:
    model = control_policy(28).to(device)

vae_state_dict = torch.load("/home/cyx/project/visual_servo/isaac/vae/model/mse_loss_1000.pth")
vae_model = vae(16, 12).to(device)
vae_model.load_state_dict(vae_state_dict)
vae_model.eval()
assets_root_path = "/home/cyx/project/visual_servo/assets"
batch_size = 128
episode = 0
if test == True:
    env = env(assets_root_path, render=True, physics_dt = 1 / 60.0)
else:
    env = env(assets_root_path, render=False, physics_dt = 1 / 60.0)
count = 0
logdir = "./isaac/logs"
if not os.path.exists(logdir):
    os.mkdir(logdir)
import datetime
time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
import tensorboardX
tensorboard_writer = tensorboardX.SummaryWriter(os.path.join(logdir, time))
min_loss = []
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def reverse_alpha_p_elu(y: torch.Tensor, alpha=1.0, eps=1e-7):
    mask = y < alpha
    if mask.ndim == 0:  # is scalar
        x = torch.log(y/alpha + eps) if mask else (y - alpha)
    else:
        x = y - alpha
        x[mask] = torch.log(y[mask] / alpha + eps)
    return x

def alpha_p_elu(x, alpha=1.0):
    return alpha + torch.nn.functional.elu(x, alpha=alpha)

while simulation_app.is_running():
    env.step()
    observation = env.get_observation()
    if observation == None:
        continue
    else:
        vel_gt, hole_desired_feature, peg_desired_feature, hole_feature_to_image, peg_feature_to_image = observation
        vae_input = np.append(hole_desired_feature, peg_desired_feature, axis=0)
        # vae_input = np.append(vae_input, peg_feature_to_image, axis=0)
        vae_input = (vae_input - np.array([320., 240.])) / np.array([640., 480.])
        vae_input = vae_input.reshape(1, -1)
        vae_input = torch.tensor(vae_input).to(device).float()
        _, _, _, latent_feature = vae_model(vae_input)
        # desired_feature = np.append(hole_desired_feature, peg_desired_feature, axis=0)
        # desired_feature = hole_desired_feature
        # desired_feature = (desired_feature - np.array([320, 240])) / np.array([640, 480])
        # desired_feature = desired_feature.reshape(1, -1)
        # desired_feature = torch.tensor(desired_feature).to(device).float()

        # current_feature = hole_feature_to_image
        current_feature = np.append(hole_feature_to_image, peg_feature_to_image, axis=0)
        current_feature = (current_feature- np.array([320, 240])) / np.array([640, 480])
        current_feature = current_feature.reshape(1, -1)
        current_feature = torch.tensor(current_feature).to(device).float()

        vel_gt = torch.tensor(vel_gt).to(device).float()
        if torch.max(torch.abs(vel_gt))  > 0.15:
            vel_gt = vel_gt / torch.max(torch.abs(vel_gt)) * 0.15
        # _, _, _, latent_feature = vae_model(desired_feature)
        input = torch.cat([latent_feature, current_feature], dim=1)
        # input = torch.cat([desired_feature, current_feature], dim=1)
        model.eval()
        vel_dir, vel_norm = model(input)
        vel = vel_dir / torch.norm(vel_dir) * alpha_p_elu(vel_norm, alpha=1.0)
        print(alpha_p_elu(vel_norm, alpha=1.0))
        print(vel)
        # vel = vel_dir / torch.norm(vel_dir) * vel_norm
        env.apply_vel(vel.cpu().detach().numpy().reshape(-1))
        if test == True:
            continue
        # collect data
        if count == 0:
            input_batch = input.unsqueeze(0)
            vel_gt_batch = vel_gt.unsqueeze(0)
        else:
            input_batch = torch.cat([input_batch, input.unsqueeze(0)], dim=0)
            vel_gt_batch = torch.cat([vel_gt_batch, vel_gt.unsqueeze(0)], dim=0)
        count += 1
        if count == batch_size:
            episode += 1
            count = 0
            model.train()
            vel_dir_batch, vel_norm_batch = model(input_batch)
            dir_loss = 1 - torch.nn.functional.cosine_similarity(vel_dir_batch, vel_gt_batch, dim=-1).mean()
            vel_gt_norm_batch = torch.norm(vel_gt_batch, dim=1, keepdim=True)
            # norm_loss = 0.5 * torch.nn.functional.mse_loss(vel_norm_batch, reverse_alpha_p_elu(vel_gt_norm_batch), reduction='mean')
            norm_loss = 0.5 * torch.nn.functional.mse_loss(vel_norm_batch, vel_gt_norm_batch, reduction='mean')
            loss = dir_loss * 1000 + norm_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tensorboard_writer.add_scalar("dir_loss", dir_loss.item(), episode)
            tensorboard_writer.add_scalar("norm_loss", norm_loss.item(), episode)
            tensorboard_writer.add_scalar("loss", loss.item(), episode)
            min_loss.append(loss.item())
            if min(min_loss) == loss.item():
                torch.save(model.state_dict(), "./isaac/model/control_policy.pth")




            
        