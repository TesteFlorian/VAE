import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torchsummary import summary

import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(
        self,
        img_size,
        nb_channels,
        z_dim,
        beta=0.1,
    ):
        """ """
        super().__init__()

        self.img_size = img_size
        self.nb_channels = nb_channels
        self.z_dim = z_dim
        self.beta = beta

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(self.nb_channels, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            # nn.Conv2d(4, 8, 3, 2, 1),
            # nn.BatchNorm2d(8),
            # nn.ReLU(),

            nn.Conv2d(8, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # nn.Conv2d(32, 64, 5, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),

            # nn.Conv2d(64, 128, 3, 2, 1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),

           

        )
        self.final_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(59392, self.z_dim * 2)
            # nn.Conv2d(16, self.z_dim * 2, kernel_size=1, stride=1, padding=0)
        )

        self.initial_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 59392),
            nn.Unflatten(1, (32, 32, 58)),
            # nn.ConvTranspose2d(self.z_dim, 16,
            #     kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(16),
            # nn.ReLU()
        )

        self.conv_decoder = nn.Sequential(

            # nn.ConvTranspose2d(128, 64, 3, 2, 1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            
            # nn.ConvTranspose2d(64, 32, 5, 2, 1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            # nn.ConvTranspose2d(8, 4, 3, 2, 1),
            # nn.BatchNorm2d(8),
            # nn.ReLU(),

            nn.ConvTranspose2d(8, self.nb_channels, 4, 2, (2,1),
                output_padding=(0, 1),dilation=3),
            nn.BatchNorm2d(self.nb_channels),
            nn.ReLU(),
        )

    def encoder(self, x):
        x = self.conv_encoder(x)
        x = self.final_encoder(x)
        # print("Shape of encoder:", x.shape)

        return x[:, :self.z_dim], x[:, self.z_dim :]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(torch.mul(logvar, 0.5))
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decoder(self, z):
        z = self.initial_decoder(z)
        x = self.conv_decoder(z)
        x = nn.Sigmoid()(x)
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        self.mu = mu
        self.logvar = logvar

        return self.decoder(z), (self.mu, self.logvar)


    def xent_continuous_ber(self, recon_x, x, gamma=1):
        # print("Shape of x:", x.shape)
        # print("Shape of recon_x:", recon_x.shape)
        """p(x_i|z_i) a continuous bernoulli"""
        eps = 1e-6

        def log_norm_const(x):
            # numerically stable computation
            x = torch.clamp(x, eps, 1 - eps)
            x = torch.where((x < 0.49) | (x > 0.51), x, 0.49 * torch.ones_like(x))
            return torch.log((2 * self.tarctanh(1 - 2 * x)) / (1 - 2 * x) + eps)

        recon_x = gamma * recon_x  # like if we multiplied the lambda ?
        return torch.mean(
            (
                x * torch.log(recon_x + eps)
                + (1 - x) * torch.log(1 - recon_x + eps)
                + log_norm_const(recon_x)
            ),
            dim=(1),  # it use to be (1)
        )

    def mean_from_lambda(self, l):
        """because the mean of a continuous bernoulli is not its lambda"""
        l = torch.clamp(l, 10e-6, 1 - 10e-6)
        l = torch.where((l < 0.49) | (l > 0.51), l, 0.49 * torch.ones_like(l))
        return l / (2 * l - 1) + 1 / (2 * self.tarctanh(1 - 2 * l))

    def kld(self):
        return 0.5 * torch.mean(
            -1
            - self.logvar
            + self.mu.pow(2)
            + self.logvar.exp(),
            dim=(1),
        )

    def tarctanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def compute_loss(self, x, recon_x):
        """
        """

        rec_term = torch.mean(
            torch.mean(self.xent_continuous_ber(recon_x, x), dim=(1, 2))
        )
        kld = torch.mean(self.kld())

        beta = 0.001
        loss = rec_term - beta * kld

        loss_dict = {
            "loss": loss,
            "rec_term": rec_term,
            "kld": kld,
            "beta*kld": beta * kld,
        }

        return loss, loss_dict

    def step(self, inputs):
        X = inputs
        rec, _ = self.forward(X)

        loss, loss_dict = self.compute_loss(X, rec)#[:, :, :256, :464], rec)

        rec = self.mean_from_lambda(rec)

        return loss, rec, loss_dict



