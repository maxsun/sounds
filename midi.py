import mido
import json
import time
import tone as T
import numpy as np
import math
import threading

note_data = json.loads(open('./notes.json').read())

midi_to_note = {}
for note in note_data:
    midi_to_note[note_data[note]['midi']] = note


mid = mido.MidiFile('./midi/DancingQueen.mid')

tone = T.Test([])
t = threading.Thread(target=tone.audio_loop)
t.start()

print('started tone loop(s)')

tone.on = True

# for note in notesdata:
#     tone.frequency = notesdata[note]['frequency']
#     time.sleep(duration)
atk = 0.15
dcy = 0.15
d = 2

start_time = time.time()
t
for i in np.linspace(440, 600, math.ceil(5 / (atk + dcy))):
    tone.frequencies = { i }
    tone.attack(atk)
    # tone.volume = 0.5
    # time.sleep(duration)
    tone.decay(dcy)

tone.on = False
tone.frequencies = set()
print('Done')
print(time.time() - start_time)
time.sleep(0.1)



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
            # P.on(midi_to_note[midnote])
            tone.frequencies.add(freq)
            tone.volume = 0.5
            tone.on = True
            tone.attack(duration)
        elif msg.type == 'note_off':
            # P.off(midi_to_note[midnote])
            if freq in tone.frequencies:
                tone.frequencies.remove(freq)
                # tone.decay(0.2)

    except AttributeError as e:
        print(e, msg)

tone.on = False

