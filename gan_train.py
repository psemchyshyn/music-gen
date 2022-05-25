from lightning.data import MusicDataWrapperCNN
from lightning.gan import MuseGAN
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml



if __name__ == "__main__":

    with open("configs/config_gan.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    dm = MusicDataWrapperCNN(config)
    lit = MuseGAN(config)


    checkpoint_callback = ModelCheckpoint(
        monitor="train_disc_loss",
        dirpath="checkpoints_gan/",
        filename="model-{epoch:02d}-{train_disc_loss:.2f}",
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        mode="min",
    )

    estopping_callback = EarlyStopping(monitor="train_disc_loss", patience=10)

    trainer = Trainer(
                    progress_bar_refresh_rate=100,
                    max_epochs=300,
                    callbacks=[checkpoint_callback, estopping_callback],
                    default_root_dir="logs_gan",
                    check_val_every_n_epoch=1000,
                    num_sanity_val_steps=0)

    trainer.fit(lit, datamodule=dm)
