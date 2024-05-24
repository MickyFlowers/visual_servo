from vae_dataset import vae_dataset
from torch.utils.data import DataLoader
from vae import vae
import torch
import os
import tensorboardX
import datetime
import tqdm

# dataset directory and file name   
root_dir = "/home/cyx/project/visual_servo/isaac/vae/data"
file_name = "desired_feature04:49PM on May 11, 2024.npy"
# device
device = "cuda" if torch.cuda.is_available() else "cpu"
# get current time
time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
# make logs directory if not exist
if not os.path.exists("isaac/vae/logs"):
    os.mkdir("logs")
# tensorboard writer
logdir = os.path.join("isaac/vae/logs", time)
tensorboard_writer = tensorboardX.SummaryWriter(logdir)

# initialize seed
seed = 42
torch.manual_seed(seed)

# create dataset and dataloader
dataset = vae_dataset(root_dir, file_name)
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = vae(16, 12).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criteria = torch.nn.MSELoss(reduction='mean')
epoches = 100
loss_lst = []
for i in tqdm.tqdm(range(epoches), desc="Training", ncols=100):
    model.train()
    print("\n")
    total_loss = 0
    total_mse_loss = 0
    total_latent_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        # normalization
        batch = (batch - torch.tensor([320, 240])) / torch.tensor([640, 480], dtype=torch.float32)
        batch = batch.reshape(batch.shape[0], -1)
        batch = batch.to(device, dtype=torch.float32)
        # print(batch.shape)
        # normal
        output, mu, log_var, _ = model(batch)
        mse_loss = criteria(output, batch)
        latent_loss = model.latent_loss(mu, log_var)
        loss = mse_loss + latent_loss * 0.

        total_loss += loss
        total_mse_loss += mse_loss
        total_latent_loss += latent_loss

        loss.backward()
        optimizer.step()
    print("total_loss: ", total_loss.item())
    print("total_mse_loss: ", total_mse_loss.item())
    print("total_latent_loss: ", total_latent_loss.item())
    tensorboard_writer.add_scalar("loss", total_loss.item(), i)
    tensorboard_writer.add_scalar("mse_loss", total_mse_loss.item(), i)
    tensorboard_writer.add_scalar("latent_loss", total_latent_loss.item(), i)
    loss_lst.append(total_loss.item())
    if min(loss_lst) == total_loss.item():
        torch.save(model.state_dict(), "./isaac/vae/model/vae.pth")
