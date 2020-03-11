import mido
import json
import time
import synth as Synth
import numpy as np
import math
import threading
import pyaudio

note_data = json.loads(open('./notes.json').read())

midi_to_note = {}
for note in note_data:
    midi_to_note[note_data[note]['midi']] = note


mid = mido.MidiFile('./midi/DancingQueen.mid')

output_io = pyaudio.PyAudio()
S = Synth.Synth(output_io)


for i, msg in enumerate(mid.play()):

    try:
        midnote = msg.note
        _type = msg.type
        duration = msg.time
        velocity = msg.velocity

        if midnote not in midi_to_note:
            continue
        freq = note_data[midi_to_note[midnote]]['frequency']
        
        if msg.type == 'note_on':
            S.play(freq)
            # P.on(midi_to_note[midnote])
            # tone.frequencies.add(freq)
            # tone.volume = 0.5
            # tone.on = True
            # tone.attack(duration)
        elif msg.type == 'note_off':
            # P.off(midi_to_note[midnote])
            S.stop(freq)
            # if freq in tone.frequencies:
            #     tone.frequencies.remove(freq)
                # tone.decay(0.2)

    except AttributeError as e:
        print(e, msg)

# tone.on = False

