#!/usr/bin/env python3
"""Create a simple test audio file for testing transcription"""

import wave
import struct
import math

# Parameters for the audio file
sample_rate = 44100  # samples per second
duration = 3  # seconds
frequency = 440  # Hz (A4 note)
amplitude = 16384  # volume (max is 32767 for 16-bit audio)

# Generate the audio data
num_samples = sample_rate * duration
samples = []

for i in range(num_samples):
    t = float(i) / sample_rate
    # Create a simple sine wave
    value = amplitude * math.sin(2 * math.pi * frequency * t)
    # Add some variation to make it more interesting
    if i % (sample_rate // 2) == 0:
        frequency = 440 if frequency == 880 else 880
    samples.append(int(value))

# Write to WAV file
output_file = "samples/test_audio.wav"
with wave.open(output_file, 'w') as wav_file:
    # Set parameters: 1 channel, 2 bytes per sample, sample rate
    wav_file.setnchannels(1)  # mono
    wav_file.setsampwidth(2)   # 2 bytes = 16 bits
    wav_file.setframerate(sample_rate)
    
    # Write the samples
    for sample in samples:
        # Pack as 16-bit signed integer
        data = struct.pack('<h', sample)
        wav_file.writeframes(data)

print(f"Created test audio file: {output_file}")
print(f"Duration: {duration} seconds")
print(f"Sample rate: {sample_rate} Hz")
print(f"File size: ~{len(samples) * 2 / 1024:.1f} KB")
