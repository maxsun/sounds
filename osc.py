'''Oscillator Synthesis Experiments'''
import math
import pyaudio
import threading
import time
import numpy as np
from matplotlib import pyplot as plt
from functools import reduce
from scipy import signal


class Osc:
    '''An Oscillator Model'''

    def __init__(self,
                 frequency: 'Osc',
                 phase: 'Osc',
                 amp: 'Osc',
                 waveform=np.sin):
        self.waveform = waveform
        self.frequency = frequency
        self.phase = phase
        self.amplitude = amp


    def sample(self, start: float, end: float, sample_rate=44100):
        '''Sample the oscillator for a portion of time'''
        duration = end - start

        freq = self.frequency.sample(start, end, sample_rate)
        phase = self.phase.sample(start, end, sample_rate)
        amp = self.amplitude.sample(start, end, sample_rate)
        phase_shift = phase * np.pi * 2

        num_samples = math.ceil(sample_rate * duration)
        sampletimes = np.linspace(
            0, duration, num_samples, endpoint=False) + start
        x = sampletimes * 2 * np.pi * freq + phase_shift
        return (amp * self.waveform(x)).astype(np.float32)


class ConstOsc(Osc):
    '''A Constant Oscillator'''

    def __init__(self, amp: float):
        # super().__init__()
        self.waveform = None
        self.frequency = None
        self.phase = None
        self.amplitude = amp

    def sample(self, start: float, end: float, sample_rate=44100):
        '''Sample the constant value'''
        return self.amplitude


class Synth:
    '''An Oscillator, attacks, decays, and multiple voices'''

    def __init__(self, output):
        self.output = output
        self.oscillators = set()
        self.isAlive = True

        self.loop_thread = threading.Thread(target=self.audio_loop, args=())
        self.loop_thread.start()
        self.time_step = 0.01

        self.attacks = {}

    def audio_loop(self):
        stream = self.output.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=44100,
                                  output=True)
        print('Starting Audio Loop..')
        t = 0
        while self.isAlive:
            to_merge = []
            for osc in self.oscillators:
                x = osc.sample(t, t + self.time_step)
                if osc in self.attacks:
                    for atk_event in self.attacks[osc]:
                        atk_start = atk_event[0]
                        atk = atk_event[1]
                        print(t, atk_start + len(atk) / 44100)
                        if t >= atk_start and t < atk_start + len(atk) / 44100 + 0.000001:
                            event_time = t - atk_start
                            atk_index = min(len(atk)-1, math.ceil(event_time * 44100))
                            print(atk[atk_index])
                            x = x * atk[int(atk_index)]

                to_merge.append(x)

            if len(to_merge) > 0:
                total = reduce(lambda a, b: a + b, to_merge)
                s = total / len(to_merge)
                stream.write((0.5 * s).tobytes())
            else:
                stream.write(np.linspace(0, 0, int(self.time_step * 44100)).tobytes())
            t += self.time_step

        print('Ending Audio Loop...')
        stream.stop_stream()
        stream.close()

    def kill(self):
        self.isAlive = False


local_linear = Osc(
    frequency=ConstOsc(1),
    waveform=lambda x: abs(signal.sawtooth(x)),
    amp=ConstOsc(440),
    phase=ConstOsc(0))


a = Osc(
    frequency=ConstOsc(440),
    waveform=np.sin,
    # amp=ConstOsc(1),
    amp=Osc(ConstOsc(0.5), phase=ConstOsc(0), amp=ConstOsc(1)),
    phase=ConstOsc(0))

b = Osc(
    frequency=ConstOsc(660),
    waveform=np.sin,
    amp=Osc(ConstOsc(0.5), phase=ConstOsc(0), amp=ConstOsc(1)),
    phase=ConstOsc(0))

output_io = pyaudio.PyAudio()

S = Synth(output_io)
S.oscillators.add(a)
time.sleep(1)
S.oscillators.add(b)

time.sleep(1)
# S.attacks[b] = (1, np.linspace(1, 1, 44100 * 1))


time.sleep(1)
S.kill()
output_io.terminate()

# Next, we want Context-Sensitive oscillators
