import pickle
import random
import os
import numpy as np

from tensorflow.keras.models import load_model
from music21 import stream, note, chord, tempo


# ==========================================
# SETTINGS
# ==========================================

MODEL_FILE = "models/music_lstm.keras"
MAPPING_FILE = "models/note_mapping.pkl"
OUTPUT_FILE = "generated_music/ai_generated_melody.mid"

SEQUENCE_LENGTH = 30
GENERATE_NOTES = 120


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading trained AI music model...")

model = load_model(MODEL_FILE)

print("AI model loaded successfully!")


# ==========================================
# LOAD NOTE MAPPING
# ==========================================

with open(MAPPING_FILE, "rb") as file:
    mapping = pickle.load(file)

note_to_int = mapping["note_to_int"]
int_to_note = mapping["int_to_note"]

print(f"Available musical events: {len(note_to_int)}")


# ==========================================
# CREATE INITIAL SEED
# ==========================================

print("Creating musical seed...")

all_notes = list(note_to_int.keys())

seed = random.sample(
    all_notes,
    min(SEQUENCE_LENGTH, len(all_notes))
)

while len(seed) < SEQUENCE_LENGTH:
    seed.append(random.choice(all_notes))


# ==========================================
# GENERATE NEW NOTES
# ==========================================

generated_notes = list(seed)

print("Generating new melody...")

for _ in range(GENERATE_NOTES):

    pattern = [
        note_to_int[value]
        for value in generated_notes[-SEQUENCE_LENGTH:]
    ]

    input_data = np.array(
        pattern,
        dtype=np.int32
    ).reshape(1, SEQUENCE_LENGTH)

    prediction = model.predict(
        input_data,
        verbose=0
    )[0]

    # Randomized sampling makes the melody less repetitive
    temperature = 1.0

    prediction = np.log(
        prediction + 1e-9
    ) / temperature

    probabilities = np.exp(prediction)

    probabilities = probabilities / np.sum(
        probabilities
    )

    index = np.random.choice(
        len(probabilities),
        p=probabilities
    )

    next_note = int_to_note[index]

    generated_notes.append(next_note)


print(f"Generated {len(generated_notes)} musical events.")


# ==========================================
# CONVERT TO MIDI
# ==========================================

print("Converting generated music to MIDI...")

music_stream = stream.Stream()

# Set tempo
music_stream.append(
    tempo.MetronomeMark(
        number=100
    )
)

for value in generated_notes:

    try:

        # Chord
        if "." in value:

            pitches = value.split(".")

            chord_notes = [
                note.Note(int(pitch))
                for pitch in pitches
            ]

            new_chord = chord.Chord(
                chord_notes
            )

            new_chord.quarterLength = 0.5

            music_stream.append(
                new_chord
            )

        # Single note
        else:

            new_note = note.Note(value)

            new_note.quarterLength = 0.5

            music_stream.append(
                new_note
            )

    except Exception:
        continue


# ==========================================
# SAVE MIDI
# ==========================================

os.makedirs(
    "generated_music",
    exist_ok=True
)

music_stream.write(
    "midi",
    fp=OUTPUT_FILE
)


print("\n================================")
print("AI MUSIC GENERATION COMPLETED!")
print("================================")

print(f"Generated MIDI saved to:")
print(OUTPUT_FILE)

print("\nYour AI has created a new melody! 🎵🤖")