from lightning.data import MusicDataWrapper
from lightning.rnn import LitAttentionRNN
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml



if __name__ == "__main__":
    seed_everything(42)

    with open("config_rnn.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    dm = MusicDataWrapper(config)
    lit = LitAttentionRNN(config, dm.num_notes_classes, dm.num_duration_classes)


    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints_classical_lstm",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    estopping_callback = EarlyStopping(monitor="val_loss", patience=2)

    trainer = Trainer(deterministic=True,
                    progress_bar_refresh_rate=100,
                    max_epochs=100,
                    callbacks=[checkpoint_callback, estopping_callback])

    trainer.fit(lit, dm)
    trainer.test(lit, dm, ckpt_path="best")
