import os
import sys
import queue
import subprocess
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper

# Path Setup to find files in the parent directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
FACE_SCRIPT = os.path.join(PARENT_DIR, "1_facemapping.py")

SAMPLE_RATE = 16000
TEMP_FILENAME = "agent1_temp.wav"

def record_audio():
    q = queue.Queue()
    def callback(indata, frames, time, status):
        if status: print(status, file=sys.stderr)
        q.put(indata.copy())

    input("Press [ENTER] to talk to Agent 1...")
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
    print("Agent 1 Ready. Commands: 'face', 'no face', 'exit'.")

    face_process = None

    try:
        while True:
            if record_audio():
                result = model.transcribe(TEMP_FILENAME, fp16=False)
                text = result["text"].lower().strip()
                print(f"\nYou said: {text}")

                # Routing Logic
                if "exit" in text:
                    print("Shutting down...")
                    if face_process: face_process.terminate()
                    break
                elif "no face" in text:
                    if face_process:
                        face_process.terminate()
                        face_process = None
                        print("Agent: Terminated face mapping.")
                    else:
                        print("Agent: Face mapping is not currently running.")
                elif "face" in text:
                    # Check if it's already running to prevent opening multiples
                    if face_process is None or face_process.poll() is not None:
                        print("Agent: Launching face mapping script...")
                        face_process = subprocess.Popen([sys.executable, FACE_SCRIPT])
                    else:
                        print("Agent: Face mapping is already running.")

                if os.path.exists(TEMP_FILENAME): os.remove(TEMP_FILENAME)
    except KeyboardInterrupt:
        if face_process: face_process.terminate()
    finally:
        if os.path.exists(TEMP_FILENAME): os.remove(TEMP_FILENAME)

if __name__ == "__main__":
    main()