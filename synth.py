
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

    def sample(self, start: float, end: float, sample_rate: int = 44100):
        n = math.floor((end - start) * sample_rate)
        return np.linspace(self.amplitude, self.amplitude, n)


class Osc:

    def __init__(self, frequency, phase, amplitude):
        self.waveform = np.sin
        self.frequency = frequency
        self.phase = phase
        self.amplitude = amplitude
        self.nickname = None

    def sample(self, start: float, end: float, sample_rate: int = 44100):

        duration = end - start
        num_samples = math.floor(duration * sample_rate)
        sampletimes = np.linspace(start, end, num_samples)

        freq = self.frequency.sample(start, end)
        phase = self.phase.sample(start, end)
        phase_shift = np.pi * 2 * phase

        amp = self.amplitude.sample(start, end)[:len(sampletimes)]
        # print('>>', np.average(amp), start, end)
        # print(len(amp), len(sampletimes))
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

    def sample(self, start: float, end: float, sample_rate: int = 44100):

        duration = end - start
        num_samples = math.ceil(duration * sample_rate)

        mod_start = max(self.start, start)
        mod_end = min(self.end, end)
        mod_duration = mod_end - mod_start

        before_val = self.before.sample(start, end)
        after_val = self.after.sample(start, end)

        # print(mod_duration)
        if mod_duration < 0:
            return np.linspace(after_val, after_val, num_samples).astype(np.float32)

        y = min(1, (start - self.start) / self.end)
        e = min(1, end / self.end)
        sampletimes = np.linspace(y, e, math.ceil(mod_duration * sample_rate))

        front_padding = np.linspace(before_val, before_val, int(
            sample_rate * max(0, self.start - start)))
        back_padding = np.linspace(after_val, after_val, int(
            sample_rate * max(0, end - self.end)))

        result = np.concatenate(
            [front_padding,
                self.expr(sampletimes),
                back_padding])

        return result.astype(np.float32)


class Synth:
    '''An Oscillator, attacks, decays, and multiple voices'''

    def __init__(self, output):
        self.output = output
        self.oscillators = set()
        self.isAlive = True

        self.loop_thread = threading.Thread(target=self.audio_loop, args=())
        self.loop_thread.start()

        self.loop_time = 0
        self.time_step = 0.01


    def audio_loop(self):
        stream = self.output.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=44100,
                                  output=True)
        print('Starting Audio Loop..')
        t = 0
        self.loop_time = 0
        while self.isAlive:
            to_merge = []
            for osc in frozenset(self.oscillators):
                x = osc.sample(t, t + self.time_step)
                to_merge.append(x)

            if len(to_merge) > 0:
                total = reduce(lambda a, b: a + b, to_merge)
                s = total / len(to_merge)
                stream.write((s).tobytes())
            else:
                stream.write(np.linspace(
                    0, 0, int(self.time_step * 44100)).tobytes())
            t += self.time_step
            self.loop_time += self.time_step

        print('Ending Audio Loop...')
        stream.stop_stream()
        stream.close()

    def kill(self):
        self.isAlive = False

    def play(self, frequency):
        print('Playing @', self.loop_time)
        lin_event = Event(
            start=self.loop_time,
            duration=0.1,
            before=Constant(0),
            after=Constant(0.5),
            expr=lambda x: x * 0.5
        )
        o = Osc(frequency=Constant(frequency),
                phase=Constant(0),
                amplitude=Constant(0.5))
        # o.events.append(lin_event)
        o.nickname = frequency
        self.oscillators.add(o)


    def stop(self, frequency):
        for osc in frozenset(self.oscillators):
            if osc.nickname == frequency:
                self.oscillators.remove(osc)



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
    start=0.1,
    duration=0.5,
    before=Constant(0),
    after=Constant(0.5),
    expr=lambda x: x * 0.5
)
sin2 = Osc(
    frequency=Constant(660),
    phase=Constant(0),
    amplitude=amp_ex2
)

# output_io = pyaudio.PyAudio()
# S = Synth(output_io)
# S.play(440)
# time.sleep(1)
# S.play(660)
# time.sleep(1)
# S.play(220)
# time.sleep(1)
# S.stop(440)
# d = 0.02
# plt.plot(amp_ex2.sample(1, 5), label='amp')
# plt.plot(sin2.sample(1, 5), label='oscillator')
# plt.legend()
# plt.show()

# stream = output_io.open(format=pyaudio.paFloat32,
#                         channels=1,
#                         rate=44100,
#                         output=True)


# print('going...')
# # plt.plot()
# stream.write(sin2.sample(1, 5).tobytes())
# # sources = [sin_a, sin2]

# time = 0
# step = 10
# to_merge = []
# for x in sources:
#     sample = x.sample(time, time + step)
#     to_merge.append(sample)

# merged = reduce(lambda a, b: a + b, to_merge) / len(to_merge)

# stream.write(merged.tobytes())
