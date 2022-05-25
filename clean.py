from music21 import converter, chord, note
import pickle
import os
import argparse
import numpy as np


def load_music(data_name, filename, n_bars, n_steps_per_bar):
    file = os.path.join(data_name, filename)
    with np.load(file, encoding='bytes', allow_pickle=True) as f:
        data = f['train']

    data_ints = []

    for x in data:
        counter = 0
        cont = True
        while cont:
            if not np.any(np.isnan(x[counter:(counter+4)])):
                cont = False
            else:
                counter += 4

        if n_bars * n_steps_per_bar < x.shape[0]:
            data_ints.append(x[counter:(counter + (n_bars * n_steps_per_bar)),:])


    data_ints = np.array(data_ints)

    n_songs = data_ints.shape[0]
    n_tracks = data_ints.shape[2]

    data_ints = data_ints.reshape([n_songs, n_bars, n_steps_per_bar, n_tracks])

    max_note = 83

    where_are_NaNs = np.isnan(data_ints)
    data_ints[where_are_NaNs] = max_note + 1
    max_note = max_note + 1

    data_ints = data_ints.astype(int)

    num_classes = max_note + 1

    
    data_binary = np.eye(num_classes)[data_ints]
    data_binary[data_binary==0] = -1
    data_binary = np.delete(data_binary, max_note,-1)

    data_binary = data_binary.transpose([0,1,2, 4,3])
    return data_binary, data_ints, data


def clean_data(path, seq_len=32,  out="./cleaned_data"):
    music_list = os.listdir(path)
    notes = []
    durations = []

    for i, file in enumerate(music_list):
        print(i+1, "Parsing %s" % file)
        original_score = converter.parse(os.path.join(path, file)).chordify()
        

        # score = original_score.transpose(interval)

        notes.extend(['START']*seq_len)
        durations.extend([0]*seq_len)

        for element in original_score.flat:
            
            if isinstance(element, note.Note):
                if element.isRest:
                    notes.append(str(element.name))
                    durations.append(element.duration.quarterLength)
                else:
                    notes.append(str(element.nameWithOctave))
                    durations.append(element.duration.quarterLength)

            if isinstance(element, chord.Chord):
                notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                durations.append(element.duration.quarterLength)

    with open(os.path.join(out, 'notes'), 'wb') as f:
        pickle.dump(notes, f)
    with open(os.path.join(out, 'durations'), 'wb') as f:
        pickle.dump(durations, f) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Music data cleaning')
    parser.add_argument("--indir", help="Input data dir")
    parser.add_argument("--outdir", help="Output data dir")
    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    clean_data(args.indir, 32, args.outdir)
