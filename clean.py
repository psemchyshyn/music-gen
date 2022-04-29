from music21 import converter, chord, note
import pickle
import os
import argparse


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
