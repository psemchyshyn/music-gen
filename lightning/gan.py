import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from models.gan import Generator, Discriminator


class MuseGAN(pl.LightningModule):
    def __init__(self, config):
        super(MuseGAN, self).__init__()

        self.lr_disc = config["model"]["lr_disc"]
        self.lr_gen = config["model"]["lr_gen"]
        self.latent_size = config["model"]["latent_size"]
        self.n_critic = config["model"]["n_critic"]
        self.b1 = config["model"]["b1"]
        self.b2 = config["model"]["b2"]
        self.lambda_gp = config["model"]["lambda_gp"]


        self.discriminator = Discriminator()
        self.generator = Generator(self.latent_size)

        
    def forward(self, latent):
        return self.generator(latent)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def generator_step(self, z):
        generated = self(z)
        loss = -torch.mean(generated)
        return loss


    def discriminator_step(self, x, z):
        # labels_valid = torch.ones(z.size(0), 1)
        # labels_fake = torch.ones(z.size(0), 0)
        # labels_valid = labels_valid.type_as(x)
        # labels_fake = labels_fake.type_as(x)


        # loss_fake = F.cross_entropy(self.discriminator(self(z)), labels_fake)
        # loss_valid = F.cross_entropy(self.discriminator(x), labels_valid)
        fake_x = self(z)
        gradient_penalty = self.compute_gradient_penalty(x.data, fake_x.data)
        loss = -torch.mean(self.discriminator(x)) + torch.mean(self.discriminator(fake_x)) + self.lambda_gp*gradient_penalty

        return loss

        
    def step(self, batch, optimizer_idx, type="train"):
        x, _ = batch

        z = torch.randn(x.size(0), self.latent_size)

        if optimizer_idx == 0:
            loss = self.generator_step(z)
            self.log(f"{type}_gen_loss", loss)
            return {f"{type}_gen_loss": loss, "loss": loss}

        else:
            loss = self.discriminator_step(x, z)
            self.log(f"{type}_disc_loss", loss)
            return {f"{type}_disc_loss": loss, "loss": loss}

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self.step(batch, optimizer_idx, "train")

    def validation_step(self, batch, batch_idx, optimizer_idx):
        return self.step(batch, optimizer_idx, "valid")

    def test_step(self, batch, batch_idx, optimizer_idx):
        return self.step(batch, optimizer_idx, "test")

    def configure_optimizers(self):

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr_gen, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_disc, betas=(self.b1, self.b2))
        return (
            {'optimizer': opt_g, 'frequency': 1},
            {'optimizer': opt_d, 'frequency': self.n_critic}
        )

    def generate(self, latent):
        rec = self(latent)
        rec = torch.argmax(rec.permute(0, 3, 1, 2), dim=3) #  32 * 4 * 83
        return rec.detach().numpy()
