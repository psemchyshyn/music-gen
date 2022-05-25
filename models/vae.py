from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_size) -> None:
        super(Encoder, self).__init__()
        # 4 x 83 x 32
        # Encoding layers
        self.enc_conv1 = nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = (4, 4), stride = (4, 4)) # 20x9
        self.enc_conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (4, 4), stride = (4, 4)) # 5x2
        self.enc_conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (5, 2)) # 1x1 

        self.enc_batch1 = nn.BatchNorm2d(64)
        self.enc_batch2 = nn.BatchNorm2d(128)
        self.enc_batch3 = nn.BatchNorm2d(256)

        self.enc_lin = nn.Linear(256, 256)
        self.enc_mu = nn.Linear(256, latent_size)
        self.enc_sigma = nn.Linear(256, latent_size)
        self.N = torch.distributions.Normal(0, 1)
        self.dropout = nn.Dropout(0.4)


    def forward(self, x):
        x = F.relu(self.enc_batch1(self.enc_conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.enc_batch2(self.enc_conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.enc_batch3(self.enc_conv3(x)))

        x = self.enc_lin(nn.Flatten()(x))
        mu = self.enc_mu(x)
        sigma = torch.exp(self.enc_sigma(x))

        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()   
        return z    


class Decoder(nn.Module):
    def __init__(self, latent_size) -> None:
        super(Decoder, self).__init__()

        self.dec_lin = nn.Linear(latent_size, 256)
        self.dec_conv1 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = (5, 2)) # 5 x 2
        self.dec_conv2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = (4, 4), stride = (4, 4), output_padding=(1, 0)) # 21x8
        self.dec_conv3 = nn.ConvTranspose2d(in_channels = 64, out_channels = 4, kernel_size = (3, 4), stride = (4, 4)) # 83*32

        self.dec_batch3 = nn.BatchNorm2d(64)
        self.dec_batch2 = nn.BatchNorm2d(128)
        self.dec_batch1 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.dec_lin(x))
        x = self.dec_batch1(nn.Unflatten(dim=1, unflattened_size=(256, 1, 1))(x))
        x = F.relu(self.dec_batch2(self.dec_conv1(x)))
        x = F.relu(self.dec_batch3(self.dec_conv2(x)))
        x = F.relu(self.dec_conv3(x))

        return x


class ConvolutionalVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(ConvolutionalVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        z = self.encoder(input)
        return self.decoder(z)
