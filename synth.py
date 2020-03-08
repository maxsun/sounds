

import numpy as np
from scipy import signal, fftpack
from scipy.fft import fftshift
from typing import NamedTuple, Callable
from functools import reduce
import math
import time
import threading
import json
import pyaudio

# p = pyaudio.PyAudio()

class Wave:
    
    # we want to describe waves as a function of time as well

    def __init__(self, frequency=440, amplitude=1, phase=0):
        self.frequency = frequency # Cycles per Second
        self.amplitude = amplitude
        self.phase = phase # Angular offset
        self.period = 1 / frequency # Seconds per cycle
        
    def sample(self, duration=1, sample_rate=44100):
        num_samples = math.ceil(sample_rate * duration)
        phase_shift = self.phase * np.pi * 2
        sampletimes =  np.linspace(0, duration, num_samples, endpoint=False)
        return (self.amplitude * np.sin(sampletimes * 2 * np.pi * self.frequency + phase_shift)).astype(np.float32)
    

def average_samples(samples):
    total = reduce(lambda a, b: a + b, samples)
    return np.array(total / len(samples)).astype(np.float32)


class Test:

    def __init__(self, p):
        # attacking is a dict mapping frequency to time remaining on the attack
        self.pyaudio = p
        self.isAlive = True
        self.sample_rate = 44100
        t = threading.Thread(target=self.audio_loop, args=())
        t.start()
        
        # a channel has a frequency, amplitude, phase
        # can be on or off, and can have effects applied that affect its samples
        self.frequencies = set()
        self.attacks = {}

    def attack(self, frequency):
        self.attacks[frequency] = 1


    def audio_loop(self):
        stream = self.pyaudio.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=self.sample_rate,
                        output=True)
        print('looping...')
        loop_duration = 0.05
        while self.isAlive:
            to_merge = []
            for freq in self.frequencies:
                w = Wave(frequency=freq)
                x = w.sample(loop_duration)

                if freq in self.attacks:
                    atk = 1 - self.attacks[freq]
                    ``

                to_merge.append(x)
            if len(to_merge) > 0:
                s = average_samples(to_merge)
                stream.write((0.5 * s).tobytes())
            else:
                time.sleep(loop_duration)
            



        print('ending audio loop')
        stream.stop_stream()
        stream.close()
        self.pyaudio.terminate()

    def kill(self):
        self.isAlive = False




t = Test(pyaudio.PyAudio())
t.frequencies.add(440)
time.sleep(1)
t.frequencies.remove(440)
t.frequencies.add(220)
time.sleep(2)
t.kill()
# p.terminate()
