from lightning.data import MusicDataWrapperCNN
from lightning.seg import LitMusicSeg
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml



if __name__ == "__main__":
    seed_everything(42)

    with open("configs/config_vae.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    dm = MusicDataWrapperCNN(config)
    lit = LitMusicSeg(config)


    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath="checkpoints_seg_cross-entropy",
        filename="model-{epoch:02d}-{valid_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    estopping_callback = EarlyStopping(monitor="valid_loss", patience=5)

    trainer = Trainer(deterministic=True,
                    progress_bar_refresh_rate=100,
                    max_epochs=100,
                    callbacks=[checkpoint_callback, estopping_callback],
                    default_root_dir="logs_seg_cross_entropy")

    trainer.fit(lit, datamodule=dm)
