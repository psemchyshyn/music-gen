import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AttentionRNN(nn.Module):
    def __init__(self, input_size_notes, input_size_durs, note_emb_size, dur_emb_size, layer_dim, hidden_size, use_lstm=False):
        super(AttentionRNN, self).__init__()

        self.layer_dim = layer_dim
        self.hidden_size = hidden_size

        self.note_emb = nn.Embedding(input_size_notes, note_emb_size)
        self.dur_emb = nn.Embedding(input_size_durs, dur_emb_size)

        self.attention_layer = nn.Linear(hidden_size, 1)

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

        # h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_size))

        output, _ = self.rnn(x)

        weights = []

        for i in range(output.size(1)):
            state = output[:, i]
            weight = self.attention_layer(state)[:, 0]
            weights.append(weight)

        weights = torch.stack(weights).transpose(1, 0)

        norm_weights = F.softmax(weights, 1)

        norm_weights = norm_weights.view(norm_weights.size(0), -1, 1)
        att_state = torch.bmm(output.transpose(1, 2), norm_weights).squeeze()

        note = self.fc_notes(att_state)
        dur = self.fc_durs(att_state)

        return note, dur
