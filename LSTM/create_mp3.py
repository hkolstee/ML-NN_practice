import numpy as np
from pypiano import Piano
from mingus.containers import Note

def main():
    piano = Piano()
    piano.play("C-4")

if __name__ == "__main__":
    main()