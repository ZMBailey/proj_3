import glob
import librosa
import librosa.display
import IPython.display
import matplotlib.pyplot as plt

def read_songs(n=None, folder=None, suffix=['mp3', 'm4a'], verbose=False):
    
    paths = []
    songs = []
    
    if folder is None:
        folder = '*.'
    else:
        folder += '/*.'
            
    for end in suffix:
        paths += glob.glob(folder + end)
    
    if n is None:
        n = len(paths)
    
    if verbose:
        print(f"Found {n} file(s) ending with {str(suffix)[1:-1]} in './{folder[:-2]}' folder.")
    
    for path in paths[:n]:
        if '/' in path:
            songname = path.rsplit('/', 1)[1][:-4]
        else:
            songname = path[:-4]
        audio, sr = librosa.load(path, res_type='kaiser_fast')
        songs.append((audio, sr, songname))
        
    return songs


def play_button(y, rate, start_t=0, stop_t=None):
    '''Insert a play button that clips the audio between start and stop times. 
    By default, play the entire audio file.'''
    start = librosa.time_to_samples(start_t)
    
    if stop_t is not None:
        stop = librosa.time_to_samples(stop_t)
        
    return IPython.display.display(IPython.display.Audio(data=y[start:stop], rate=rate))

    
def chromaplot(y, rate, start_t=0, stop_t=None, play=True, harmonic_input=False):
    
    start = librosa.time_to_samples(start_t)

    if stop_t is not None:
        stop = librosa.time_to_samples(stop_t)
    
    if harmonic_input is False:
        h, p = librosa.effects.hpss(y[start:stop])
    else:
        h = y[start:stop]
        
    C = librosa.feature.chroma_cqt(y=h, sr=rate)
    
    plt.figure(figsize=(12,4))
    librosa.display.specshow(C, sr=rate, x_axis='time', y_axis='chroma', vmin=0, vmax=1)

    plt.title('Chromagram')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    if play:
        return play_button(y, rate, start_t, stop_t)
    

    
    

