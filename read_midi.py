import mido
import json
import time
import synth as S
import numpy as np
import math
import threading

note_data = json.loads(open('./notes.json').read())

midi_to_note = {}
for note in note_data:
    midi_to_note[note_data[note]['midi']] = note


mid = mido.MidiFile('./midi/DancingQueen.mid')

note_samples = {}
for note in note_data.keys():
    freq = note_data[note]['frequency']
    w = S.Wave(freq=freq, func=S.signal.square)
    w2 = S.Wave(freq=freq, func=np.sin)
    w3 = S.Wave(freq=freq*2)

    waves = [w, w2, w3]

    note_samples[note] = S.average_samples([x.sample(duration=0.1) for x in waves])

P = S.Player(note_samples)
P.start(0.2)

for i, msg in enumerate(mid.play()):

    try:
        midnote = msg.note
        _type = msg.type
        duration = msg.time
        velocity = msg.velocity

        if midnote not in midi_to_note:
            continue
        
        if msg.type == 'note_on':
            P.on(midi_to_note[midnote])
        elif msg.type == 'note_off':
            P.off(midi_to_note[midnote])

    except AttributeError as e:
        print(e, msg)

