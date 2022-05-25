import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, MeanSquaredError, F1Score
from models.seg import MusicSeg


class LitMusicSeg(pl.LightningModule):
    def __init__(self, config):
        super(LitMusicSeg, self).__init__()
        self.lr = config["model"]["lr"]

        self.model = MusicSeg()

        metrics = MetricCollection([MeanSquaredError()])
        self.metrics = {"train": metrics.clone("train"), "valid": metrics.clone("valid"), "test": metrics.clone("test")}
        
    def forward(self, x):
        return self.model(x)

    def step(self, batch, type):
        x, y = batch

        reconstructed_x = self(x)

        reconstructed_x = reconstructed_x.permute(0, 2, 1, 3)
        y = y.permute(0, 2, 1, 3)

        loss = F.cross_entropy(reconstructed_x, torch.argmax(y, dim=1))
        # loss = F.mse_loss(reconstructed_x, y)
        self.log_dict({f"{type}_loss": loss, **self.metrics[type](reconstructed_x, y)})

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

    def generate(self, start):
        y = self.model(start)
        y = torch.argmax(y.permute(0, 3, 1, 2), dim=3) #  32 * 4 * 83
        return y.detach().numpy()
