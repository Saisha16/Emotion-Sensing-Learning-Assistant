import sounddevice as sd
from scipy.io.wavfile import write, read

def record_audio(duration=5, fs=44100, filename='audio.wav'):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("Recording saved as", filename)
    
    # Play back the recording
    print("Playing back...")
    fs, data = read(filename)
    sd.play(data, fs)
    sd.wait()  # Wait until playback is finished

# Usage
record_audio()
