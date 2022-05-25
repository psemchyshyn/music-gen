import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from lightning.dataset import MusicDataset, MusicDatasetExt, MusicDatasetCNN

class MusicDataWrapper(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.path = config["data"]["path"]
        self.batch_size = config["data"]["batch_size"]
        self.seq_len = config["data"]["seq_len"]
        self.is_generation = config["data"]["is_generation"]

        if self.is_generation:
            self.dataset = MusicDatasetExt(self.path, self.seq_len)
        else:
            self.dataset = MusicDataset(self.path, self.seq_len)

        size = len(self.dataset)
        self.ds_train, self.ds_val, self.ds_test = random_split(self.dataset, [int(0.6*size), int(0.2*size), size - int(0.6*size) - int(0.2*size)])

        self.num_notes_classes = self.dataset.num_notes_classes
        self.num_duration_classes =  self.dataset.num_duration_classes

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, pin_memory=True)

class MusicDataWrapperCNN(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.path = config["data"]["path"]
        self.batch_size = config["data"]["batch_size"]
        self.seq_len = config["data"]["seq_len"]

        self.ds_train = MusicDatasetCNN(self.path, "train", self.seq_len)
        self.ds_val = MusicDatasetCNN(self.path, "valid", self.seq_len)
        self.ds_test = MusicDatasetCNN(self.path, "test", self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, pin_memory=True)
