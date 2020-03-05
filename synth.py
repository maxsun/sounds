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
import logging

p = pyaudio.PyAudio()


class Sample(NamedTuple):
    rate: int
    duration: float
    data: np.array

    def __add__(self, other: 'Sample') -> 'Sample':
        if isinstance(other, Sample) and self.rate == other.rate:
            return Sample(self.rate, self.duration, np.add(self.data, other.data))
        else:
            raise('Error adding Samples!')

    def __sub__(self, other: 'Sample') -> 'Sample':
        if isinstance(other, Sample) and self.rate == other.rate:
            return Sample(self.rate, self.duration, np.subtract(self.data, other.data))
        else:
            raise Exception('Error subtracting Samples!')


    def subsample(self, start_time, end_time):
        a = math.floor(start_time * self.rate)
        b = math.ceil(end_time * self.rate)
        return Sample(self.rate, end_time - start_time, self.data[a:b])

    def display(self, sample_rate=44100, duration=1):
        # period = 1 / self.freq
        # samples = self.sample(sample_rate, duration)
        # samples = samples[:int(len(samples) * period) + 1]
        samples = self.data
        plot.plot(np.linspace(0, self.duration, len(samples)), samples)
        
        plot.title('Sqaure wave - 5 Hz sampled at 1000 Hz /second')

        plot.xlabel('Time')
        plot.ylabel('Amplitude')
        plot.grid(True, which='both')

        plot.axhline(y=0, color='k')
        plot.ylim(-1.5, 1.5)

        plot.show()

    def spectrogram(self, sample_rate=44100, duration=1):
        f, t, Sxx = signal.spectrogram(self.data, self.rate, return_onesided=False)
        plot.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0))
        plot.ylabel('Frequency [Hz]')
        plot.ylim(0, 10000)
        plot.xlabel('Time [sec]')
        plot.show()


class Wave(NamedTuple):
    amplitude: int = 1
    phase: float = 0
    freq: float = 440
    func: Callable[[np.array], np.array] = np.sin

    def sample(self, sample_rate=44100, duration=1) -> np.array:
        points = np.linspace(0, duration, math.ceil(sample_rate * duration))
        samples = self.amplitude * self.func(np.pi * 2 * points * self.freq + (self.phase * np.pi * 2))
        return samples.astype(np.float32)

    def period(self):
        return 1 / self.freq



stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                output=True)


def play_sample(sample: np.array, n: int=1, volume=0.5):
    '''Plays a <sample> of 1 period of a wave <n> times.'''
    for _ in range(n):
        stream.write((volume * sample).tobytes())



def average_samples(samples):
    total = reduce(lambda a, b: a + b, samples)
    return np.array(total / len(samples)).astype(np.float32)


def decay(sample: np.array, decay_time: float=1, sample_rate=44100):
    sample_duration = len(sample) / sample_rate
    for i in np.linspace(0, 1, math.ceil(decay_time / sample_duration)):
        v = 0.5 * (i - 1) ** 2
        v = min(v, 1)
        play_sample(sample, volume=v)


class Player:

    def __init__(self, samples):
        self.note_samples = samples
        self.on_notes = set()

    def on(self, note):
        self.on_notes.add(note)

    def off(self, note):
        if note in self.on_notes:
            # decay(self.note_samples[note], 0.5)
            self.on_notes.remove(note)

    def emit(self, duration=0.2, sample_rate=44100):
        while True:
            samples = []
            for note in frozenset(self.on_notes):
                samples.append(self.note_samples[note])
            if len(samples) > 0:
                w = average_samples(samples)
                play_sample(w, int(duration * sample_rate / len(w)))

    def start(self, duration):
        t = threading.Thread(target=self.emit, args=(duration, ))
        t.start()


import json
notedata = json.loads(open('notes.json').read())

note_samples = {}
for note in notedata.keys():
    freq = notedata[note]['frequency']
    w = Wave(freq=freq, func=np.sin)
    note_samples[note] = w.sample(duration=0.1)

ply = Player(note_samples)

chord1 = average_samples([
    note_samples['A4'],
    note_samples['E3'],
    note_samples['C3']
])

chord2 = average_samples([
    note_samples['G2'],
    note_samples['D3'],
    note_samples['C5']
])

chord3 = average_samples([
    note_samples['G3'],

    note_samples['A2']
])


# P = Player(note_samples)
# P.on('A4')
# P.on('E3')
# P.start()
# time.sleep(2)
# print('Turning off..')
# P.off('A4')

# time.sleep(2)
# P.off('E3')
# P.on('C2')

# time.sleep(1)
# P.off('C2')