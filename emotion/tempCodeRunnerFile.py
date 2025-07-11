def record_audio(duration=5, fs=44100, filename='audio.wav'):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("Recording saved as", filename)
