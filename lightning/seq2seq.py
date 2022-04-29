import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, F1Score
from models.seq2seq import Seq2Seq, Encoder, Decoder


class LitSeq2Seq(pl.LightningModule):
    def __init__(self, config, input_note_size, input_dur_size):
        super(LitSeq2Seq, self).__init__()
        self.input_note_size = input_note_size
        self.input_dur_size = input_dur_size
        self.lr = config["model"]["lr"]
        self.temperature = config["model"]["sampling_temperature"]


        self.encoder = Encoder(
                        input_note_size,
                        input_dur_size,
                        config["model"]["encoder"]["note_emb_size"],
                        config["model"]["encoder"]["dur_emb_size"],
                        config["model"]["encoder"]["num_recurrent_layers"],
                        config["model"]["encoder"]["hidden_size"],
                        config["model"]["encoder"]["use_lstm"]
                        )
        self.decoder = Decoder(
                        input_note_size,
                        input_dur_size,
                        config["model"]["decoder"]["note_emb_size"],
                        config["model"]["decoder"]["dur_emb_size"],
                        config["model"]["decoder"]["num_recurrent_layers"],
                        config["model"]["decoder"]["hidden_size"],
                        config["model"]["decoder"]["use_lstm"]
        )
        self.model = Seq2Seq(self.encoder, self.decoder)

        metrics = MetricCollection([Accuracy(), Precision(), Recall(), F1Score()])
        self.train_notes_metrics = metrics.clone(prefix='train_notes_')
        self.train_durs_metrics = metrics.clone(prefix="train_durs_")
        self.val_notes_metrics = metrics.clone(prefix='val_notes_')
        self.val_durs_metrics = metrics.clone(prefix="val_durs_")
        self.test_notes_metrics = metrics.clone(prefix='test_notes_')
        self.test_durs_metrics = metrics.clone(prefix="test_durs_")
        
    def forward(self, encoder_inputs, decoder_inputs=None, use_teacher_forcing=False, generate_size=None):
        return self.model(encoder_inputs, decoder_inputs, use_teacher_forcing, generate_size)

    def step(self, batch, type):
        x_note, x_dur = batch

        ds_tokenizer = self.trainer.datamodule.dataset
        # print("HERE", ds_tokenizer)

        start_note = torch.full((x_note.size(0), 1), ds_tokenizer.notes_to_tokens['START'])
        end_note = torch.full((x_note.size(0), 1), ds_tokenizer.notes_to_tokens['END'])

        start_dur = torch.full((x_dur.size(0), 1), ds_tokenizer.durations_to_tokens[0])
        end_dur = torch.full((x_dur.size(0), 1), ds_tokenizer.durations_to_tokens['END'])

        

        # print(f"HERE: {y_note.shape}")


        x_hat_note, x_hat_dur = self((x_note, x_dur), (torch.cat((start_note, x_note), dim=1), torch.cat((start_dur, x_dur), dim=1)), type == "train")
        loss = 0
        x_note = torch.cat((x_note, end_note), dim=1)
        x_dur = torch.cat((x_dur, end_dur), dim=1)
        for i in range(x_hat_note.size(1)):
            loss += F.cross_entropy(x_hat_note[:, i], x_note[:, i]) + F.cross_entropy(x_hat_dur[:, i], x_dur[:, i])

        self.log(f"{type}_loss", loss)

        # if type == "train":
        #     metrics_output_notes = self.train_notes_metrics(preds_notes, y_note)
        #     metrics_output_durs = self.train_durs_metrics(preds_durs, y_dur)
        #     self.log_dict({'train_loss': loss, **metrics_output_notes, **metrics_output_durs})
        # elif type == "val":
        #     metrics_output_notes = self.val_notes_metrics(preds_notes, y_note)
        #     metrics_output_durs = self.val_durs_metrics(preds_durs, y_dur)
        #     self.log_dict({'val_loss': loss, **metrics_output_notes, **metrics_output_durs})
        # else:
        #     metrics_output_notes = self.test_notes_metrics(preds_notes, y_note)
        #     metrics_output_durs = self.test_durs_metrics(preds_durs, y_dur)
        #     self.log_dict({'test_loss': loss, **metrics_output_notes, **metrics_output_durs})
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

        target = (torch.tensor([ds_tokenizer.notes_to_tokens["START"]]).unsqueeze(dim=0), torch.tensor([ds_tokenizer.durations_to_tokens[0]]).unsqueeze(0))
        notes_dist, durs_dist = self((tokenized_notes.unsqueeze(dim=0), tokenized_durs.unsqueeze(dim=0)), target, use_teacher_forcing=False, generate_size=seq_len)

        preds_notes = self.sample_with_temp(notes_dist.squeeze(dim=0))
        preds_durs = self.sample_with_temp(durs_dist.squeeze(dim=0))

        for note, dur in zip(preds_notes, preds_durs):

            notes.append(ds_tokenizer.tokens_to_notes[note.item()])
            durs.append(ds_tokenizer.tokens_to_durations[dur.item()])

        return notes, durs
