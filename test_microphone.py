
import sounddevice as sd
import numpy as np

duration = 5
sample_rate = 16000

print("Recording for 5 seconds... Speak now!")

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="float32"
)

sd.wait()

rms = float(np.sqrt(np.mean(audio ** 2)))

print("Recording finished.")
print(f"Captured samples: {len(audio)}")
print(f"RMS energy: {rms:.6f}")

if rms > 0.01:
    print("Sound detected in the recording.")
else:
    print("Very low sound detected. Check microphone input or volume.")
