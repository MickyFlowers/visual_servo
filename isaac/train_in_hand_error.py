from env.gripper_only_without_sim_env_peg import env
from control_policy.control_policy import control_policy
from control_policy.control_policy_dataset import control_policy_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch

eva_env = env()
train_env = env()
train_env.reset()
eva_env.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = control_policy(32).to(device)
data_size = 1e4
epoches = 1
batch_size = 64
episode = 0
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
import os
logdir = "./isaac/logs"
if not os.path.exists(logdir):
    os.mkdir(logdir)
import datetime
time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
import tensorboardX
tensorboard_writer = tensorboardX.SummaryWriter(os.path.join(logdir, time))
max_success_rate = []
count = 0

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

# fig = plt.figure()
# plt.ion()
while True:
    observation = train_env.get_observation()
    if observation == False or observation == True:
        train_env.reset()
        continue
    vel_gt, hole_desired_feature, peg_desired_feature, hole_feature_to_image, peg_feature_to_image = observation
    # fig.clf()
    # plt.plot(hole_feature_to_image[:, 0], hole_feature_to_image[:, 1], 'ro')
    # plt.ylim(0, 480)
    # plt.xlim(0, 640)
    # plt.pause(0.01)
    # desired_feature = hole_desired_feature
    desired_feature = np.append(hole_desired_feature, peg_desired_feature, axis=0)
    desired_feature = (desired_feature - np.array([320., 240.])) / np.array([640., 480.])
    desired_feature = desired_feature.reshape(1, -1)
    desired_feature = torch.tensor(desired_feature).to(device).float()

    # current_feature = hole_feature_to_image
    current_feature = np.append(hole_feature_to_image, peg_feature_to_image, axis=0)
    current_feature = (current_feature- np.array([320., 240.])) / np.array([640., 480.])
    current_feature = current_feature.reshape(1, -1)
    current_feature = torch.tensor(current_feature).to(device).float()
    vel_gt = torch.tensor(vel_gt).to(device).float()
    if torch.max(torch.abs(vel_gt))  > 0.15:
        vel_gt = vel_gt / torch.max(torch.abs(vel_gt)) * 0.15
    # _, _, _, latent_feature = vae_model(desired_feature)
    # input = torch.cat([latent_feature, current_feature], dim=1)
    input = torch.cat([desired_feature, current_feature], dim=1).squeeze(0)
    model.eval()
    vel_dir, vel_norm = model(input)
    vel = vel_dir / torch.norm(vel_dir) * alpha_p_elu(vel_norm)
    # vel = vel_dir / torch.norm(vel_dir) * vel_norm
    train_env.apply_vel(vel.cpu().detach().numpy().reshape(-1))
    # collect data
    if count == 0:
        input_batch = input.unsqueeze(0)
        vel_gt_batch = vel_gt.unsqueeze(0)
    else:
        input_batch = torch.cat([input_batch, input.unsqueeze(0)], dim=0)
        vel_gt_batch = torch.cat([vel_gt_batch, vel_gt.unsqueeze(0)], dim=0)
    count += 1
    if count % data_size == 0:
        # count = 0
        dataset = control_policy_dataset(input_batch, vel_gt_batch)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for i in range(epoches):
            total_loss = 0
            total_dir_loss = 0
            total_norm_loss = 0
            times = 0
            for batch in train_loader:
                times += 1
                model.train()
                optimizer.zero_grad()
                vel_dir_batch, vel_norm_batch = model(batch[0])
                dir_loss = 1 - torch.nn.functional.cosine_similarity(vel_dir_batch, batch[1], dim=-1).mean()
                vel_gt_norm_batch = torch.norm(batch[1], dim=1, keepdim=True)
                norm_loss = 0.5 * torch.nn.functional.mse_loss(vel_norm_batch, reverse_alpha_p_elu(vel_gt_norm_batch), reduction='mean')
                loss = 0.2 * dir_loss + norm_loss
                loss.backward()
                optimizer.step()
                total_loss += loss
                total_dir_loss += dir_loss
                total_norm_loss += norm_loss
            total_loss /= times
            total_dir_loss /= times
            total_norm_loss /= times
            episode += 1
            tensorboard_writer.add_scalar("dir_loss", total_dir_loss.item(), episode)
            tensorboard_writer.add_scalar("norm_loss", total_norm_loss.item(), episode)
            tensorboard_writer.add_scalar("loss", total_loss.item(), episode)
            # evaluation
            eva_count = 0
            success = 0
            while(eva_count < 100):
                eva_observation = eva_env.get_observation()
                if eva_observation == False:
                    eva_count += 1
                    eva_env.reset()
                    continue
                elif eva_observation == True:
                    success += 1
                    eva_count += 1
                    eva_env.reset()
                    continue
                else:
                    vel_gt, hole_desired_feature, peg_desired_feature, hole_feature_to_image, peg_feature_to_image = eva_observation
                    # desired_feature = hole_desired_feature
                    desired_feature = np.append(hole_desired_feature, peg_desired_feature, axis=0)
                    desired_feature = (desired_feature - np.array([320., 240.])) / np.array([640., 480.])
                    desired_feature = desired_feature.reshape(1, -1)
                    desired_feature = torch.tensor(desired_feature).to(device).float()
                    # current_feature = hole_feature_to_image
                    current_feature = np.append(hole_feature_to_image, peg_feature_to_image, axis=0)
                    current_feature = (current_feature- np.array([320., 240.])) / np.array([640., 480.])
                    current_feature = current_feature.reshape(1, -1)
                    current_feature = torch.tensor(current_feature).to(device).float()
                    input = torch.cat([desired_feature, current_feature], dim=1).squeeze(0)
                    model.eval()
                    vel_dir, vel_norm = model(input)
                    vel = vel_dir / torch.norm(vel_dir) * vel_norm
                    eva_env.apply_vel(vel.cpu().detach().numpy().reshape(-1))
            success_rate = success / 100.0
            tensorboard_writer.add_scalar("success_rate", success_rate, episode)
            print("Training episode: {}, loss: {}, dir_loss: {}, norm_loss: {}, success_rate: {}".format(episode, total_loss.item(), total_dir_loss.item(), total_norm_loss.item(), success_rate))
            max_success_rate.append(success_rate)
            if max(max_success_rate) == success_rate:
                torch.save(model.state_dict(), "./isaac/model/control_policy.pth")





    