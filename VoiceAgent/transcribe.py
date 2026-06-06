import sys
import os
import queue
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper

SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
TEMP_FILENAME = "temp_recording.wav"

def load_whisper_model():
    print("Loading Whisper model ('base')... This may take a moment on first run.")
    # 'base' balances speed and accuracy nicely on your i7 processor
    return whisper.load_model("base")

def record_audio_until_keypress():
    print("\nPress [ENTER] to START recording...")
    input()

    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    # Start the recording stream
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, dtype='int16')
    with stream:
        print("Recording... Press [ENTER] again to STOP.")
        input()

    audio_data = []
    while not q.empty():
        audio_data.append(q.get())

    if len(audio_data) == 0:
        print("No audio captured.")
        return False

    # Concatenate all chunks and save to a temporary WAV file
    audio_np = np.concatenate(audio_data, axis=0)
    wav.write(TEMP_FILENAME, SAMPLE_RATE, audio_np)
    return True

def main():
    model = load_whisper_model()
    print("System Ready!")

    try:
        while True:
            if record_audio_until_keypress():
                print("Transcribing...")
                # fp16=False prevents CPU warning alerts
                result = model.transcribe(TEMP_FILENAME, fp16=False) 
                text = result["text"].strip()

                print("\n--- Transcription ---")
                print(text)
                print("---------------------\n")

                if os.path.exists(TEMP_FILENAME):
                    os.remove(TEMP_FILENAME)

            print("Want to go again? (Ctrl+C to exit)")

    except KeyboardInterrupt:
        print("\nExiting transcriber application.")
        if os.path.exists(TEMP_FILENAME):
            os.remove(TEMP_FILENAME)

if __name__ == "__main__":
    main()