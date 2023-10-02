import torch
import torch.nn as nn
import torch.nn.functional as torch_func


class TorchFullyConnected(nn.Module):
    def __init__(self, input_shape, number_of_class):
        super(TorchFullyConnected, self).__init__()
        self.dens1 = nn.Linear(input_shape, 400)
        self.drop1 = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(400)

        self.dens2 = nn.Linear(400, 200)
        self.drop2 = nn.Dropout(p=0.5)
        # self.bn2 = nn.BatchNorm1d(200)
        self.last = nn.Linear(200, number_of_class)

    def forward(self, x):
        out = torch_func.selu(self.dens1(x))
        out = self.drop1(out)
        out = self.bn1(out)

        out = torch_func.selu(self.dens2(out))
        out = self.drop2(out)
        out = torch.sigmoid(self.last(out))

        return out


class TorchAttentionBase(nn.Module):
    def __init__(self, embed_dim, num_heads, number_of_class=1):
        super(TorchAttentionBase, self).__init__()
        self.embed_dim = embed_dim
        self.multi_head_attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.2, batch_first=True)

        self.dens = nn.Linear(73, 200)
        self.drop = nn.Dropout(p=0.5)
        # self.bn2 = nn.BatchNorm1d(200)
        self.last = nn.Linear(200, number_of_class)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=-1)
        x = x.repeat(1, 1, self.embed_dim)
        attn_output, attn_output_weights = self.multi_head_attn(x, x, x)
        attn_output = torch.mean(attn_output, dim=-1)

        out = torch_func.selu(self.dens(attn_output))
        out = self.drop(out)
        out = torch.sigmoid(self.last(out))
        return out


class Autoencoder(nn.Module):
    def __init__(self, feature_size):
        super(Autoencoder, self).__init__()

        self.feature_size = feature_size

        self.encoder = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            # nn.ReLU(True),
            nn.SELU(),
            nn.Linear(128, 64),
            # nn.ReLU(True),
            nn.SELU(),
            nn.Linear(64, 12),
            nn.SELU())
        # nn.ReLU(True),
        # nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            # nn.Linear(3, 12),
            # nn.ReLU(True),
            nn.Linear(12, 64),
            # nn.ReLU(True),
            nn.SELU(),
            nn.Linear(64, 128),
            # nn.ReLU(True),
            nn.SELU(),
            nn.Linear(128, self.feature_size),
            # nn.ReLU())
            nn.SELU())

    def forward(self, input_features):
        latent = self.encoder(input_features)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


# %% For dead

class DenseAutoencoder(nn.Module):
    def __init__(self, feature_size):
        super(DenseAutoencoder, self).__init__()

        self.feature_size = feature_size

        self.encoder = nn.Sequential(
            nn.Linear(self.feature_size, 64),
            # nn.ReLU(True),
            nn.SELU(),
            nn.Linear(64, 32),
            # nn.ReLU(True),
            nn.SELU(),
            nn.Linear(32, 12),
            nn.SELU())
        # nn.ReLU(True),
        # nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            # nn.Linear(3, 12),
            # nn.ReLU(True),
            nn.Linear(12, 32),
            # nn.ReLU(True),
            nn.SELU(),
            nn.Linear(32, 64),
            # nn.ReLU(True),
            nn.SELU(),
            nn.Linear(64, self.feature_size),
            # nn.ReLU()
            nn.SELU())

    def forward(self, input_features):
        latent = self.encoder(input_features)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


def re_parameterize(z_mean, z_log_var):
    eps = (torch.randn(z_mean.size(0), z_mean.size(1), device='cuda').float())

    z = z_mean + (eps * torch.exp(z_log_var / 2.))
    return z


class tt_autoencoder_trip(nn.Module):
    def __init__(self, feature_size):
        super(tt_autoencoder_trip, self).__init__()

        self.feature_size = feature_size

        self.encoder = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            # nn.Dropout(p=0.5),
            # nn.ReLU(True),
            nn.SELU(True),
            nn.Linear(128, 64),
            # nn.Dropout(p=0.5),
            # nn.ReLU(True),
            nn.SELU(True),
            nn.Linear(64, 32),
            # nn.Dropout(p=0.5),
            # nn.ReLU(True),
            nn.SELU(True))

        self.z_mean = nn.Linear(32, 10)
        self.z_log_var = nn.Linear(32, 10)

        self.decoder = nn.Sequential(

            nn.Linear(10, 32),
            nn.SELU(True),
            nn.Linear(32, 64),
            nn.SELU(True),
            nn.Linear(64, 128),
            nn.SELU(True),
            nn.Linear(128, self.feature_size),
            nn.SELU(True))

    def encoding(self, input_features):
        features = self.encoder(input_features)
        z_mean, z_log_var = self.z_mean(features), self.z_log_var(features)
        latent = re_parameterize(z_mean, z_log_var)
        return latent, z_mean, z_log_var

    def forward(self, input_features):
        latent, z_mean, z_log_var = self.encoding(input_features)
        reconstruction = self.decoder(latent)
        return reconstruction, latent, z_mean, z_log_var

        # x_anch, x_pos, x_neg = triplter(outp, y)
        # return outp, x_anch, x_pos, \
        #        x_neg, reps, z_mean, z_log_var


if __name__ == '__main__':
    model = TorchAttentionBase(embed_dim=4, num_heads=2)

    x = torch.rand(200, 73)

    t = model(x)
