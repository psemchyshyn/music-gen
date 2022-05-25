from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os


class MusicDataset(Dataset):
    def __init__(self, path, seq_len=32):
        self.path = path
        self.seq_len = seq_len
        self.prepare_data(path)

    def __getitem__(self, idx):
        tokenized_notes_x = torch.tensor([self.notes_to_tokens[i] for i in self.notes_x[idx]])
        tokenized_durs_x = torch.tensor([self.durations_to_tokens[i] for i in self.durations_x[idx]])
        tokenized_notes_y = self.notes_to_tokens[self.notes_y[idx]]
        tokenized_durs_y = self.durations_to_tokens[self.durations_y[idx]]
        return tokenized_notes_x, tokenized_notes_y, tokenized_durs_x, tokenized_durs_y

    def __len__(self):
        return len(self.notes_x)

    def prepare_data(self, path):
        with open(os.path.join(path, 'notes'), 'rb') as f:
            notes = np.array(pickle.load(f))
        with open(os.path.join(path, 'durations'), 'rb') as f:
            durations = np.array(pickle.load(f))
        
        self.notes_to_tokens = {j:i for i, j in enumerate(set(notes))}
        self.tokens_to_notes = {j:i for i, j in self.notes_to_tokens.items()}

        self.durations_to_tokens = {j:i for i, j in enumerate(set(durations))}
        self.tokens_to_durations = {j:i for i, j in self.durations_to_tokens.items()}

        self.num_duration_classes = len(self.durations_to_tokens)
        self.num_notes_classes = len(self.notes_to_tokens)

        i = 0

        self.notes_x = []
        self.durations_x = []
        self.notes_y = []
        self.durations_y = []

        while i < len(notes) - self.seq_len - 1:
            note_last = notes[i + self.seq_len]
            dur_last = durations[i + self.seq_len]
            if note_last != "START" and dur_last != 0:
                self.notes_x.append(notes[i: i + self.seq_len])
                self.durations_x.append(durations[i: i + self.seq_len])
                self.notes_y.append(note_last)
                self.durations_y.append(dur_last)

                i += 1
            else:
                i += self.seq_len


class MusicDatasetExt(MusicDataset):
    def __init__(self, path, seq_len=32):
        super().__init__(path, seq_len)

    def __getitem__(self, idx):
        tokenized_notes_x = torch.tensor([self.notes_to_tokens[i] for i in self.notes_x[idx]])
        tokenized_durs_x = torch.tensor([self.durations_to_tokens[i] for i in self.durations_x[idx]])
        return tokenized_notes_x, tokenized_durs_x

    def prepare_data(self, path):
        with open(os.path.join(path, 'notes'), 'rb') as f:
            notes = np.array(pickle.load(f))
        with open(os.path.join(path, 'durations'), 'rb') as f:
            durations = np.array(pickle.load(f))
        
        self.notes_to_tokens = {j:i for i, j in enumerate(set(notes) | {'END'})}
        self.tokens_to_notes = {j:i for i, j in self.notes_to_tokens.items()}

        self.durations_to_tokens = {j:i for i, j in enumerate(set(durations) | {'END'})}
        self.tokens_to_durations = {j:i for i, j in self.durations_to_tokens.items()}

        self.num_duration_classes = len(self.durations_to_tokens)
        self.num_notes_classes = len(self.notes_to_tokens)

        i = 0

        self.notes_x = []
        self.durations_x = []
        self.notes_y = []
        self.durations_y = []

        while i < len(notes) - self.seq_len:
            note_last = notes[i + self.seq_len - 1]
            dur_last = durations[i + self.seq_len - 1]
            if note_last != "START" and dur_last != 0:
                self.notes_x.append(notes[i: i + self.seq_len])
                self.durations_x.append(durations[i: i + self.seq_len])
                i += 1
            else:
                i += self.seq_len


class MusicDatasetCNN(Dataset):
    def __init__(self, path, mode="train", seq_len=32):
        self.path = path
        self.seq_len = seq_len
        self.mode = mode
        self.rest_idx = 82
        self.num_classes = self.rest_idx + 1
        self.prepare_data(path, mode)

    def __len__(self):
        return len(self.notes_x)

    def __getitem__(self, idx):
        notes_x = self.notes_x[idx]
        notes_y = self.notes_y[idx]
        
        notes_x = torch.stack(tuple([torch.nn.functional.one_hot(torch.tensor(note).long(), self.num_classes) for note in notes_x]))
        notes_y = torch.stack(tuple([torch.nn.functional.one_hot(torch.tensor(note).long(), self.num_classes) for note in notes_y]))

        return notes_x.permute(1, 2, 0).float(), notes_y.permute(1, 2, 0).float()

    def prepare_data(self, path, mode):
        with np.load(path, encoding='bytes', allow_pickle=True) as f:
            self.data = f[mode]

        i = 0

        self.notes_x = []
        self.notes_y = []

        for song in self.data:
            is_nan = np.where(np.isnan(song))
            song[is_nan] = self.rest_idx
            song = song.astype(np.uint8)

            while i < len(song) - 2*self.seq_len:
                self.notes_x.append(song[i: i + self.seq_len])
                self.notes_y.append(song[i + self.seq_len: i + 2*self.seq_len])
                i += 1
