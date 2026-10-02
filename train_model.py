import pickle
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# SETTINGS
# ==========================================

NOTES_FILE = "models/notes.pkl"
MODEL_FILE = "models/music_lstm.keras"
MAPPING_FILE = "models/note_mapping.pkl"

SEQUENCE_LENGTH = 30

# Limit the amount of training data for faster training
MAX_SEQUENCES = 12000

EPOCHS = 5
BATCH_SIZE = 128


# ==========================================
# LOAD PROCESSED MUSIC
# ==========================================

print("Loading processed music data...")

with open(NOTES_FILE, "rb") as file:
    notes = pickle.load(file)

print(f"Total musical events: {len(notes)}")


# ==========================================
# CREATE NOTE MAPPING
# ==========================================

unique_notes = sorted(set(notes))

print(f"Unique musical events: {len(unique_notes)}")

note_to_int = {
    value: index
    for index, value in enumerate(unique_notes)
}

int_to_note = {
    index: value
    for value, index in note_to_int.items()
}


# ==========================================
# CREATE TRAINING SEQUENCES
# ==========================================

print("Creating training sequences...")

available_sequences = len(notes) - SEQUENCE_LENGTH

if available_sequences <= 0:
    raise ValueError(
        "Not enough musical events to create training sequences."
    )

sequence_count = min(
    available_sequences,
    MAX_SEQUENCES
)

network_input = []
network_output = []

for i in range(sequence_count):

    sequence = notes[i:i + SEQUENCE_LENGTH]

    target = notes[i + SEQUENCE_LENGTH]

    network_input.append(
        [note_to_int[value] for value in sequence]
    )

    network_output.append(
        note_to_int[target]
    )


network_input = np.array(
    network_input,
    dtype=np.int32
)

network_output = np.array(
    network_output,
    dtype=np.int32
)

print(f"Training sequences used: {len(network_input)}")


# ==========================================
# BUILD FAST LSTM MODEL
# ==========================================

print("\nBuilding LSTM model...")

model = Sequential([
    
    Input(shape=(SEQUENCE_LENGTH,)),

    Embedding(
        input_dim=len(unique_notes),
        output_dim=64
    ),

    LSTM(
        128,
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(
        len(unique_notes),
        activation="softmax"
    )
])


# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("LSTM model created successfully!")


# ==========================================
# TRAIN
# ==========================================

print("\n================================")
print("STARTING FAST LSTM TRAINING")
print("================================")

early_stopping = EarlyStopping(
    monitor="loss",
    patience=2,
    restore_best_weights=True
)

model.fit(
    network_input,
    network_output,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

model.save(MODEL_FILE)

with open(MAPPING_FILE, "wb") as file:
    pickle.dump(
        {
            "note_to_int": note_to_int,
            "int_to_note": int_to_note
        },
        file
    )


# ==========================================
# COMPLETION MESSAGE
# ==========================================

print("\n================================")
print("LSTM TRAINING COMPLETED!")
print("================================")

print(f"Model saved to: {MODEL_FILE}")
print(f"Mapping saved to: {MAPPING_FILE}")

print("\nYour AI music model is ready! 🎵🤖")