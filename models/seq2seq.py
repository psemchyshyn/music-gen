import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size_notes, input_size_durs, note_emb_size, dur_emb_size, layer_dim, hidden_size, use_lstm=False):
        super(Encoder, self).__init__()

        self.layer_dim = layer_dim
        self.hidden_size = hidden_size

        self.note_emb = nn.Embedding(input_size_notes, note_emb_size)
        self.dur_emb = nn.Embedding(input_size_durs, dur_emb_size)

        if use_lstm:
            self.rnn = nn.LSTM(note_emb_size + dur_emb_size, hidden_size, layer_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(note_emb_size + dur_emb_size, hidden_size, layer_dim, batch_first=True)

        self.fc_notes = nn.Linear(hidden_size, input_size_notes)
        self.fc_durs = nn.Linear(hidden_size, input_size_durs)

    def forward(self, notes, durs):
        notes = self.note_emb(notes)
        durs = self.dur_emb(durs)
        
        x = torch.cat((notes, durs), dim=2)
        _, hn = self.rnn(x)

        # weights = []

        # for i in range(output.size(1)):
        #     state = output[:, i]
        #     weight = self.attention_layer(state)[:, 0]
        #     weights.append(weight)

        # weights = torch.stack(weights).transpose(1, 0)

        # norm_weights = F.softmax(weights, 1)

        # norm_weights = norm_weights.view(norm_weights.size(0), -1, 1)
        # att_state = torch.bmm(output.transpose(1, 2), norm_weights).squeeze()

        return hn

class Decoder(nn.Module):
    def __init__(self, input_size_notes, input_size_durs, note_emb_size, dur_emb_size, layer_dim, hidden_size, use_lstm=False):
        super(Decoder, self).__init__()

        self.layer_dim = layer_dim
        self.hidden_size = hidden_size

        self.note_emb = nn.Embedding(input_size_notes, note_emb_size)
        self.dur_emb = nn.Embedding(input_size_durs, dur_emb_size)

        # self.attention_layer = nn.Linear(hidden_size, 1)

        if use_lstm:
            self.rnn = nn.LSTM(note_emb_size + dur_emb_size, hidden_size, layer_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(note_emb_size + dur_emb_size, hidden_size, layer_dim, batch_first=True)

        self.fc_notes = nn.Linear(hidden_size, input_size_notes)
        self.fc_durs = nn.Linear(hidden_size, input_size_durs)

    def forward(self, hn, note, dur):
        note = self.note_emb(note)
        dur = self.dur_emb(dur)
        
        x = torch.cat((note, dur), dim=2)
        outputs, hidden = self.rnn(x, hn)

        note_pred = self.fc_notes(outputs)
        dur_pred = self.fc_durs(outputs)

        return note_pred, dur_pred, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, encoder_inputs, decoder_inputs=None, use_teacher_forcing=False, generate_size=None):
        if generate_size is None:
            generate_size = encoder_inputs[0].size(1)

        hidden_state = self.encoder(*encoder_inputs)

        decoder_output_notes = []
        decoder_output_durs = []

        if use_teacher_forcing:
            decoder_inputs_notes, decoder_inputs_durs = decoder_inputs
            for i in range(decoder_inputs_notes.size(1)):
                decoder_input_notes = decoder_inputs_notes[:, i].unsqueeze(1)
                decoder_input_durs = decoder_inputs_durs[:, i].unsqueeze(1)
                decoder_output_note, decoder_output_dur, hidden_state = self.decoder(hidden_state, decoder_input_notes, decoder_input_durs)

                decoder_output_notes.append(decoder_output_note)
                decoder_output_durs.append(decoder_output_dur)

        else:
            decoder_input_note = decoder_inputs[0][:, 0].unsqueeze(1)
            decoder_input_dur = decoder_inputs[1][:, 0].unsqueeze(1)
            for i in range(generate_size):

                decoder_output_note, decoder_output_dur, hidden_state = self.decoder(hidden_state, decoder_input_note, decoder_input_dur)

                decoder_input_note = torch.argmax(decoder_output_note, dim=2)
                decoder_input_dur = torch.argmax(decoder_output_dur, dim=2)
                
                decoder_output_notes.append(decoder_output_note)
                decoder_output_durs.append(decoder_output_dur)      

        decoder_output_notes = torch.cat(decoder_output_notes, dim=1)
        decoder_output_durs = torch.cat(decoder_output_durs, dim=1)

        # print("Decoder output shape", decoder_output_notes.shape)

        return decoder_output_notes, decoder_output_durs

        



