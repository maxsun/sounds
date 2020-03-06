import pyaudio
import numpy as np
import matplotlib.pyplot as plot
from scipy import signal
from scipy.fft import fftshift
from typing import NamedTuple, Callable
from functools import reduce
import math
import time
import threading
import json

p = pyaudio.PyAudio()

sample_rate = 44100
duration = 0.05
amplitude = 1
# frequency = 440
phase = 0


def sample(curr_time: float, frequency: int):
    timepoints = np.linspace(0, duration, int(sample_rate * duration))
    timepoints += curr_time
    samples = amplitude * np.sin(np.pi * 2 * timepoints * frequency + (phase * np.pi * 2))
    x = samples.astype(np.float32)
    return x


class Test:

    def __init__(self, f):
        self.frequency = f
        self.on = False
        self.volume = 0.5

    def audio_loop(self):
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=44100,
                        output=True)

        i = 0
        while True:
            if self.on:
                x = sample(i, self.frequency)
                stream.write((x * self.volume).tobytes())
            # else:
                # time.sleep(duration)
            i += duration

    def decay(self, decay_time: float=0.5, sample_rate=44100):
        x = sample(0, self.frequency)
        sample_duration = len(x) / sample_rate
        for i in np.linspace(1, 0, math.ceil(decay_time / duration)):
            v = 0.5 * (i - 1) ** 2
            self.volume = 0.5 - min(v, 1)
            print(self.volume)
            time.sleep(duration)



notesdata = json.loads(open('notes.json', 'r').read())

# def repeat(sample: np.array, n: int):
    # return np.concatenate([sample for _ in range(n)])

tone = Test(440)
t = threading.Thread(target=tone.audio_loop)
t.start()

# tone2 = Test(660)
# t2 = threading.Thread(target=tone2.audio_loop)
# t2.start()

print('started tone loop(s)')

tone.on = True

# for note in notesdata:
#     tone.frequency = notesdata[note]['frequency']
#     time.sleep(duration)

for i in range(400, 600):
    time.sleep(duration)
    tone.decay(1)
    tone.frequency = i
    tone.volume = 0.5
# time.sleep(1)
# tone.decay()

# time.sleep(2)


# tone.on = False

# tone.play(440, 1)

# play_sample(repeat(x, 20))
# sample = x
# sample_duration = len(sample) / sample_rate


# for i in np.linspace(0, 1, sample_rate * duration):
#     v = 0.5 * (i - 1) ** 2
#     v = min(v, 1)
#     stream.write((sample * volume).tobytes())


# stream.stop_stream()

# stream.close()

p.terminate()
