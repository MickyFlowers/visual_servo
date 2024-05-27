import torch.nn as nn
import torch
import numpy as np
class vae_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(vae_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)
    

class vae_decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(vae_decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return torch.tanh(self.decoder(x))

class vae(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(vae, self).__init__()
        self.encoder = vae_encoder(input_dim, 16, 16)
        self.decoder = vae_decoder(latent_dim, 16, input_dim)
        self._enc_mu = nn.Linear(16, latent_dim)
        self._enc_log_sigma = nn.Linear(16, latent_dim)
    
    def _sample_latent(self, h_enc):
        mu = self._enc_mu(h_enc)
        log_var = self._enc_log_sigma(h_enc)
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        self.mu = mu
        self.log_var = log_var

        return mu + std * eps, mu, log_var
    
    def get_mean_sample(self, x, k=1):
        lst = []
        h_enc = self.encoder(x)
        if k:
            for _ in range(k):
                lst.append(self._sample_latent(h_enc).cpu().detach().numpy())
            lst = np.array(lst)
            return torch.FloatTensor(np.mean(lst, axis=0))
        else:
            self._sample_latent(h_enc)
            return self._enc_mu(h_enc)
        
    def forward(self, x):
        h_enc = self.encoder(x)
        z, mu, log_var = self._sample_latent(h_enc)
        return self.decoder(z), mu, log_var, z
    
    def latent_loss(self, mu, log_var):
        return 0.5 * torch.mean(mu.pow(2) + log_var.exp() - log_var - 1)
    
    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))