import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection, MeanSquaredError
from models.vae import ConvolutionalVAE, Encoder, Decoder


class LitVAE(pl.LightningModule):
    def __init__(self, config):
        super(LitVAE, self).__init__()
        self.lr = config["model"]["lr"]

        self.encoder = Encoder(config["model"]["latent_size"])
        self.decoder = Decoder(config["model"]["latent_size"])
        self.model = ConvolutionalVAE(self.encoder, self.decoder)

        metrics = MetricCollection([MeanSquaredError()])
        self.metrics = {"train": metrics.clone("train"), "valid": metrics.clone("valid"), "test": metrics.clone("test")}
        
    def forward(self, x):
        return self.model(x)

    def step(self, batch, type):
        x, _ = batch

        reconstructed_x = self(x)

        reconstructed_x = reconstructed_x.permute(0, 2, 1, 3)
        x = x.permute(0, 2, 1, 3)

        loss = F.cross_entropy(reconstructed_x, torch.argmax(x, dim=1)) + self.model.encoder.kl
        self.log_dict({f"{type}_loss": loss, **self.metrics[type](reconstructed_x, x)})

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def generate(self, latent):
        rec = self.model.decoder(latent)
        rec = torch.argmax(rec.permute(0, 3, 1, 2), dim=3) #  32 * 4 * 83
        return rec.detach().numpy()
