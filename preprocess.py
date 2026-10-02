from music21 import converter, instrument, note, chord
import glob
import pickle
import os

MIDI_FOLDER = "dataset/midi"
OUTPUT_FILE = "models/notes.pkl"

os.makedirs("models", exist_ok=True)

notes = []

midi_files = (
    glob.glob(os.path.join(MIDI_FOLDER, "*.mid"))
    + glob.glob(os.path.join(MIDI_FOLDER, "*.midi"))
)

print(f"Found {len(midi_files)} MIDI files.")

for index, file in enumerate(midi_files, start=1):
    print(f"Processing {index}/{len(midi_files)}: {os.path.basename(file)}")

    try:
        midi = converter.parse(file)

        parts = instrument.partitionByInstrument(midi)

        if parts:
            elements = parts.parts[0].recurse()
        else:
            elements = midi.flat.notes

        for element in elements:

            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                chord_notes = ".".join(str(n) for n in element.normalOrder)
                notes.append(chord_notes)

    except Exception as error:
        print(f"Skipped file: {file}")
        print(f"Reason: {error}")

with open(OUTPUT_FILE, "wb") as file:
    pickle.dump(notes, file)

print("\n--------------------------------")
print("PREPROCESSING COMPLETED!")
print("--------------------------------")
print(f"Total musical events: {len(notes)}")
print(f"Saved to: {OUTPUT_FILE}")