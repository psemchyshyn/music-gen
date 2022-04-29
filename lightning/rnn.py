import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, F1Score
from models import AttentionRNN


class LitAttentionRNN(pl.LightningModule):
    def __init__(self, config, input_note_size, input_dur_size):
        super(LitAttentionRNN, self).__init__()
        self.input_note_size = input_note_size
        self.input_dur_size = input_dur_size
        self.lr = config["model"]["lr"]
        self.temperature = config["model"]["sampling_temperature"]


        self.model = AttentionRNN(input_note_size, input_dur_size, 
                                config["model"]["note_emb_size"],
                                config["model"]["dur_emb_size"],
                                config["model"]["num_recurrent_layers"],
                                config["model"]["hidden_size"],
                                config["model"]["use_lstm"]
                                )

        metrics = MetricCollection([Accuracy(), Precision(), Recall(), F1Score()])
        self.train_notes_metrics = metrics.clone(prefix='train_notes_')
        self.train_durs_metrics = metrics.clone(prefix="train_durs_")
        self.val_notes_metrics = metrics.clone(prefix='val_notes_')
        self.val_durs_metrics = metrics.clone(prefix="val_durs_")
        self.test_notes_metrics = metrics.clone(prefix='test_notes_')
        self.test_durs_metrics = metrics.clone(prefix="test_durs_")
        
    def forward(self, notes, durs):
        return self.model(notes, durs)

    def step(self, batch, type):
        x_note, y_note, x_dur, y_dur = batch
        y_hat_note, y_hat_dur = self(x_note, x_dur)
        loss = F.cross_entropy(y_hat_note, y_note) + F.cross_entropy(y_hat_dur, y_dur)

        preds_notes = self.sample_with_temp(y_hat_note)
        preds_durs = self.sample_with_temp(y_hat_dur)

        if type == "train":
            metrics_output_notes = self.train_notes_metrics(preds_notes, y_note)
            metrics_output_durs = self.train_durs_metrics(preds_durs, y_dur)
            self.log_dict({'train_loss': loss, **metrics_output_notes, **metrics_output_durs})
        elif type == "val":
            metrics_output_notes = self.val_notes_metrics(preds_notes, y_note)
            metrics_output_durs = self.val_durs_metrics(preds_durs, y_dur)
            self.log_dict({'val_loss': loss, **metrics_output_notes, **metrics_output_durs})
        else:
            metrics_output_notes = self.test_notes_metrics(preds_notes, y_note)
            metrics_output_durs = self.test_durs_metrics(preds_durs, y_dur)
            self.log_dict({'test_loss': loss, **metrics_output_notes, **metrics_output_durs})
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def sample_with_temp(self, preds):
        if self.temperature == 0:
            return torch.max(preds, dim=1)[1]
        else:
            preds = torch.log(preds) / self.temperature
            exp_preds = torch.exp(preds)
            preds = exp_preds / torch.sum(exp_preds, dim=1)
            return torch.multinomial(preds, 1).squeeze()

    def generate(self, ds_tokenizer, start_seq=(["START"]*32, [0]*32), seq_len=32, temperature=None):
        if temperature is None:
            temperature = self.temperature

        notes, durs = start_seq

        tokenized_notes = torch.tensor([ds_tokenizer.notes_to_tokens[i] for i in notes])
        tokenized_durs = torch.tensor([ds_tokenizer.durations_to_tokens[i] for i in durs])

        notes = []
        durs = []

        for _ in range(seq_len):
            notes_dist, durs_dist = self(tokenized_notes.unsqueeze(dim=0), tokenized_durs.unsqueeze(dim=0))
            preds_note = self.sample_with_temp(notes_dist.unsqueeze(dim=0)).squeeze().item()
            preds_dur = self.sample_with_temp(durs_dist.unsqueeze(dim=0)).squeeze().item()

            
            tokenized_notes = torch.cat([tokenized_notes, torch.tensor([preds_note])])[1:]
            tokenized_durs = torch.cat([tokenized_durs, torch.tensor([preds_dur])])[1:]

            notes.append(ds_tokenizer.tokens_to_notes[preds_note])
            durs.append(ds_tokenizer.tokens_to_durations[preds_dur])

        return notes, durs
        



