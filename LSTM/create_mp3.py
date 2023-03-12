import numpy as np
from mingus.containers import Note
from mingus.midi import fluidsynth
import mingus.core.notes as notes
from mingus.containers import NoteContainer, Bar, Track

# initialize soundfont file that will be used to play music
fluidsynth.init("YamahaC5Grand-v2_4.sf2", "alsa")

def main():
    # load network output music
    voices = np.loadtxt("output/output.txt")
    voices = np.loadtxt("input.txt")
    # print(voices.shape)
    voices = voices[2000:]
    
    encoded_voices = [Track(), Track(), Track(), Track()] 
    for row, notes in enumerate(voices):
        noteContainer = NoteContainer()
        for voice, note in enumerate(notes):
            if note:
                # count the repetitions
                count = 0
                while(True):
                    if (voices[row + count][voice] == note):
                        count += 1
                    else:
                        break
                
                # create note
                n = Note()
                n.from_int(int(note))
                b = Bar()
                b.place_notes(n, 16)
                encoded_voices[voice].add_bar(b)
            else:
                b = Bar()
                b.place_rest(16)
                encoded_voices[voice].add_bar(b)
                
    # print(np.shape(encoded_voices))
    fluidsynth.play_Tracks(encoded_voices, [1,2,3,4], 120)

if __name__ == "__main__":
    main()