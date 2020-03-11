
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fftpack
from scipy.fft import fftshift
from typing import NamedTuple, Callable
from functools import reduce
import math
import time
import threading
import json
import pyaudio


class Constant:
    
    def __init__(self, value):
        self.amplitude = value
        
    def sample(self, start: float, end: float, sample_rate: int=44100):
        n = math.ceil((end - start) * sample_rate)
        return self.amplitude
    

class Osc:
    
    def __init__(self, frequency, phase, amplitude):
        self.waveform = np.sin
        self.frequency = frequency
        self.phase = phase
        self.amplitude = amplitude
        
    def sample(self, start: float, end: float, sample_rate: int=44100):
        
        duration = end - start
        num_samples = math.floor(duration * sample_rate)
        sampletimes = np.linspace(start, end, num_samples)
        
        freq = self.frequency.sample(start, end)
        phase = self.phase.sample(start, end)
        phase_shift = np.pi * 2 * phase
        
        amp = self.amplitude.sample(start, end)[:len(sampletimes)]
        print(len(amp), len(sampletimes))
        x = amp * np.sin(np.pi * 2 * freq * sampletimes + phase_shift)
        
        results = x
        return results.astype(np.float32)
    

class Event:
    
    def __init__(self, expr, start: float, duration: float, before, after):
        self.start = start
        self.duration = duration
        self.end = start + duration
        self.expr = expr
        self.before = before
        self.after = after
        
    def sample(self, start: float, end: float, sample_rate: int=44100):
        
        duration = end - start
        num_samples = math.ceil(duration * sample_rate)
        
        mod_start = max(self.start, start)
        mod_end = min(self.end, end)
        mod_duration = mod_end - mod_start

        
        print('Mod Start:', mod_start)
        print('Mod End:', mod_end)
        print('Mod Duration:', mod_end)

        before_val = self.before.sample(start, end)
        after_val = self.after.sample(start, end)


        if mod_duration < 0:
            return np.linspace(before_val, before_val, num_samples).astype(np.float32)
        sampletimes = np.linspace(0, 1, math.ceil(mod_duration * sample_rate))
        
        front_padding = np.linspace(before_val, before_val, int(sample_rate * max(0, self.start - start)))
        back_padding = np.linspace(after_val, after_val, int(sample_rate * max(0, end - self.end)))
        print('Padding:', len(front_padding), 'back:', len(back_padding))

        result = np.concatenate(
            [front_padding,
                self.expr(sampletimes),
                back_padding])

        return result.astype(np.float32)


amp_ex = Event(
    start=0,
    duration=1,
    before=Constant(0),
    after=Constant(0.5),
    expr=lambda x: x
)

sin_a = Osc(
    frequency=Constant(440),
    phase=Constant(0),
    amplitude=amp_ex
)

amp_ex2 = Event(
    start=0.5,
    duration=1,
    before=Constant(0),
    after=Constant(0.5),
    expr=lambda x: x
)
sin2 = Osc(
    frequency=Constant(660),
    phase=Constant(0),
    amplitude=amp_ex2
)

output_io = pyaudio.PyAudio()
# d = 0.02
plt.plot(amp_ex.sample(0, 3), label='amp')
# plt.plot(sin2.sample(0.25, 1), label='oscillator')
# plt.legend()
plt.show()

stream = output_io.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=44100,
                        output=True)


sources = [sin_a, sin2]

time = 0
step = 10
to_merge = []
for x in sources:
    sample = x.sample(time, time + step)
    to_merge.append(sample)

merged = reduce(lambda a, b: a + b, to_merge) / len(to_merge)

stream.write(merged.tobytes())
