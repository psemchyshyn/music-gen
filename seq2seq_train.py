from lightning.data import MusicDataWrapper
from lightning.seq2seq import LitSeq2Seq
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml



if __name__ == "__main__":
    seed_everything(42)

    with open("config_seq2seq.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    dm = MusicDataWrapper(config)
    lit = LitSeq2Seq(config, dm.num_notes_classes, dm.num_duration_classes)


    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints_seq2seq",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    estopping_callback = EarlyStopping(monitor="val_loss", patience=2)

    trainer = Trainer(deterministic=True,
                    progress_bar_refresh_rate=100,
                    max_epochs=100,
                    callbacks=[checkpoint_callback, estopping_callback],
                    default_root_dir="logs_seq2seq")

    trainer.fit(lit, datamodule=dm)
    trainer.test(lit, dm, ckpt_path="best")
