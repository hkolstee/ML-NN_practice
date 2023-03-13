import numpy as np
from mingus.containers import Note
from mingus.midi import fluidsynth
import mingus.core.notes as notes
from mingus.containers import NoteContainer, Bar, Track, Composition
from mingus.midi import midi_file_out

# initialize soundfont file that will be used to play music
fluidsynth.init("soundfonts/YamahaC5Grand-v2_4.sf2", "alsa")
# fluidsynth.init("soundfonts/Cello_Maximus.sf2", "alsa")
# fluidsynth.init("soundfonts/040_Florestan_String_Quartet.sf2", "alsa")

def main():
    # load network output music
    voices = np.loadtxt("output/output.txt")
    voices = np.loadtxt("input.txt")
    # print(voices.shape)
    # voices = voices[]
    
    encoded_voices = [Track(), Track(), Track(), Track()] 
    
    for i, notes in enumerate([voices[:,0], voices[:,1], voices[:,2], voices[:,3]]):
        # initialize as impossible note
        last_note = -1
        count = 1
        for j, note in enumerate(notes):
            if note:
                if ((note == last_note) or (j == 0)):
                    # same note as previous note
                    count += 1
                    last_note = note
                    
                    if (j + count > len(notes)):
                        # current note reaches end of file
                        n = Note()
                        n.from_int(int(last_note))
                        b = Bar()
                        b.place_notes(n, 16/count)
                        encoded_voices[i].add_bar(b)
                else:
                    # different note encountered
                    # add previous note with its duration to track
                    n = Note()
                    n.from_int(int(last_note))
                    b = Bar()
                    
                    # 8 should be 1/2 -> 2
                    # 16 should be 1 -> 1
                    # 32 should be 2 -> 0.5
                    b.place_notes(n, duration = 16/count)
                    encoded_voices[i].add_bar(b)
                    
                    # reset
                    count = 1
                    last_note = note
            else:
                # current note = 0, means a pause (silence)
                b = Bar()
                b.place_rest(16)
                encoded_voices[i].add_bar(b)

    composition = Composition()
    composition.add_track(encoded_voices[0])
    composition.add_track(encoded_voices[1])
    composition.add_track(encoded_voices[2])
    composition.add_track(encoded_voices[3])
    
    midi_file_out.write_Composition("output.midi", composition)

    # fluidsynth.play_Track(encoded_voices[2])
    
if __name__ == "__main__":
    main()