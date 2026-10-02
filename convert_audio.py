import music21
import numpy as np
import wave
import os

MIDI_FILE = "generated_music/ai_generated_melody.mid"
WAV_FILE = "generated_music/ai_generated_melody.wav"

SAMPLE_RATE = 44100


def note_frequency(note_name):
    return float(note_name.pitch.frequency)


def create_audio():

    print("Converting AI melody to audio...")

    score = music21.converter.parse(MIDI_FILE)

    notes = score.flatten().notes

    if not notes:
        print("No notes found.")
        return

    total_duration = max(
        float(n.offset + n.duration.quarterLength)
        for n in notes
    )

    audio = np.zeros(
        int(total_duration * 0.5 * SAMPLE_RATE) + SAMPLE_RATE,
        dtype=np.float32
    )

    for element in notes:

        start = float(element.offset) * 0.5
        duration = float(element.duration.quarterLength) * 0.5

        start_sample = int(start * SAMPLE_RATE)
        end_sample = min(
            start_sample + int(duration * SAMPLE_RATE),
            len(audio)
        )

        if end_sample <= start_sample:
            continue

        length = end_sample - start_sample

        if isinstance(element, music21.note.Note):

            frequency = note_frequency(element)

            t = np.arange(length) / SAMPLE_RATE

            sound = 0.25 * np.sin(
                2 * np.pi * frequency * t
            )

            audio[start_sample:end_sample] += sound

        elif isinstance(element, music21.chord.Chord):

            frequencies = [
                note_frequency(n)
                for n in element.notes
            ]

            t = np.arange(length) / SAMPLE_RATE

            sound = np.zeros(length)

            for frequency in frequencies:
                sound += np.sin(
                    2 * np.pi * frequency * t
                )

            if len(frequencies) > 0:
                sound = (
                    sound / len(frequencies)
                ) * 0.25

            audio[start_sample:end_sample] += sound

    audio = np.clip(audio, -1, 1)

    audio_data = (
        audio * 32767
    ).astype(np.int16)

    os.makedirs("generated_music", exist_ok=True)

    with wave.open(WAV_FILE, "wb") as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(
            audio_data.tobytes()
        )

    print()
    print("==============================")
    print("AUDIO GENERATION COMPLETED!")
    print("==============================")
    print("Audio saved to:")
    print(WAV_FILE)


if __name__ == "__main__":
    create_audio()