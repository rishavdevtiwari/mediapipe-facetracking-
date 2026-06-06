import os
import sys
import queue
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper

SAMPLE_RATE = 16000
TEMP_FILENAME = "demo_temp.wav"

def record_audio():
    q = queue.Queue()
    def callback(indata, frames, time, status):
        if status: print(status, file=sys.stderr)
        q.put(indata.copy())

    input("Press [ENTER] to talk to the Demo Agent...")
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, dtype='int16')
    with stream:
        print("Listening... Press [ENTER] to stop.")
        input()

    audio_data = []
    while not q.empty(): audio_data.append(q.get())
    if not audio_data: return False

    wav.write(TEMP_FILENAME, SAMPLE_RATE, np.concatenate(audio_data, axis=0))
    return True

def main():
    print("Loading model...")
    model = whisper.load_model("base")
    print("Demo Agent Ready. Say 'high', 'hi', or 'exit'.")

    try:
        while True:
            if record_audio():
                result = model.transcribe(TEMP_FILENAME, fp16=False)
                text = result["text"].lower().strip()
                print(f"\nYou said: {text}")

                if "exit" in text:
                    print("Exiting Demo Agent.")
                    break
                elif "high" in text:
                    print("Agent: meow")
                elif "hi" in text:
                                        print("Agent: meow")
                else:
                    print("Agent: I am waiting for 'high' or 'hi'.")

                if os.path.exists(TEMP_FILENAME): os.remove(TEMP_FILENAME)
    except KeyboardInterrupt:
        pass
    finally:
        if os.path.exists(TEMP_FILENAME): os.remove(TEMP_FILENAME)

if __name__ == "__main__":
    main()