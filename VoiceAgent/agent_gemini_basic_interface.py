import os
import sys
import queue
import threading
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper
from google import genai
from google.genai import types
from dotenv import load_dotenv
from gtts import gTTS
import pygame
import tkinter as tk
from tkinter import scrolledtext

# -SYSTEM CONFIg
SAMPLE_RATE = 16000
TEMP_FILENAME = "voice_input.wav"
AUDIO_RESPONSE = "voice_output.mp3"



load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY in environment configuration.")

client = genai.Client(api_key=api_key)
config = types.GenerateContentConfig(
    system_instruction="You are a helpful, concise AI voice assistant. Keep responses strictly under 2 sentences."
)
chat_session = client.chats.create(model='gemini-2.5-flash', config=config)

# Thread communication
audio_queue = queue.Queue()
recording_stream = None
is_recording = False
whisper_model = None

class VoiceAgentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gemini Voice Agent UI")
        self.root.geometry("520x420")
        
        # Status Bar
        self.status_label = tk.Label(root, text="Loading Whisper Engine...", font=("Arial", 11, "bold"), fg="orange")
        self.status_label.pack(pady=10)
        
        # Output Terminal Log
        self.log_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=15, width=58)
        self.log_area.pack(pady=10)
        self.log_area.insert(tk.END, "System: Booting modules...\n")
        self.log_area.configure(state='disabled')
        
        # Core Controls
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        self.mic_btn = tk.Button(btn_frame, text="🎙️ Mic (Talk)", command=self.start_recording, width=12, state='disabled')
        self.mic_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="🛑 Stop", command=self.stop_recording, width=12, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        self.exit_btn = tk.Button(btn_frame, text="❌ Exit", command=self.exit_program, width=12)
        self.exit_btn.grid(row=0, column=2, padx=5)
        
        # Load heavy deep learning models in a background thread
        threading.Thread(target=self.init_whisper, daemon=True).start()

    def update_log(self, text):
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, text + "\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state='disabled')

    def update_status(self, text, color="black"):
        self.status_label.config(text=text, fg=color)

    def init_whisper(self):
        global whisper_model
        whisper_model = whisper.load_model("base")
        self.update_status("Agent Online & Ready", "green")
        self.update_log("System: Whisper localized successfully. Press 'Mic' to speak.")
        self.mic_btn.config(state='normal')

    def start_recording(self):
        global is_recording, recording_stream, audio_queue
        if is_recording: return
        
        is_recording = True
        self.mic_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.update_status("Listening to microphone input...", "red")
        
        while not audio_queue.empty():
            audio_queue.get()
            
        def audio_callback(indata, frames, time, status):
            audio_queue.put(indata.copy())

        recording_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, dtype='int16')
        recording_stream.start()

    def stop_recording(self):
        global is_recording, recording_stream
        if not is_recording: return
            
        is_recording = False
        self.stop_btn.config(state='disabled')
        self.update_status("Processing capture buffer...", "blue")
        
        recording_stream.stop()
        recording_stream.close()
        
        threading.Thread(target=self.process_audio, daemon=True).start()

    def process_audio(self):
        try:
            audio_data = []
            while not audio_queue.empty():
                audio_data.append(audio_queue.get())
                
            if not audio_data:
                self.reset_ui_state()
                return

            wav.write(TEMP_FILENAME, SAMPLE_RATE, np.concatenate(audio_data, axis=0))
            
            self.update_status("Running speech transcription...", "purple")
            result = whisper_model.transcribe(TEMP_FILENAME, fp16=False)
            user_text = result["text"].strip()
            
            if not user_text:
                self.reset_ui_state()
                return
                
            self.update_log(f"You: {user_text}")

            # Send string execution payload to active Gemini session
            self.update_status("Evaluating response sequence...", "darkblue")
            gemini_response = chat_session.send_message(user_text)
            clean_response = gemini_response.text.replace("*", "")
            
            self.update_log(f"Agent: {clean_response}")
            
            self.update_status("Synthesizing audio feedback...", "darkgreen")
            self.speak(clean_response)
            
        except Exception as e:
            self.update_log(f"Pipeline Exception: {e}")
        finally:
            if os.path.exists(TEMP_FILENAME):
                os.remove(TEMP_FILENAME)
            self.reset_ui_state()

    def reset_ui_state(self):
        self.update_status("Agent Online & Ready", "green")
        self.mic_btn.config(state='normal')

    def speak(self, text):
        try:
            tts = gTTS(text=text, lang='en', tld='com')
            tts.save(AUDIO_RESPONSE)
            
            pygame.mixer.init()
            pygame.mixer.music.load(AUDIO_RESPONSE)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            pygame.mixer.quit()
            if os.path.exists(AUDIO_RESPONSE):
                os.remove(AUDIO_RESPONSE)
        except Exception as e:
            print(f"Audio Driver Error: {e}")

    def exit_program(self):
        global is_recording, recording_stream
        self.update_status("Terminating components...", "black")
        if is_recording and recording_stream:
            recording_stream.stop()
            recording_stream.close()
            
        if os.path.exists(TEMP_FILENAME): os.remove(TEMP_FILENAME)
        if os.path.exists(AUDIO_RESPONSE): os.remove(AUDIO_RESPONSE)
            
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAgentGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.exit_program)
    root.mainloop()