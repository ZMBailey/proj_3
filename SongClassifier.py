from __future__ import unicode_literals
import youtube_dl
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import librosa
import numpy as np
import pandas as pd

class SongClassifier():
    
    genres = ['Country',
        'Jazz',
        'Metal',
        'Hip_Hop',
        'Electronic',
        'Classical']
    
    def __init__(self):
        df = pd.read_csv('data_3k.csv')
        genre_map = {g:i for i, g in enumerate(df.label.unique())}
        #df = df[df.label != 'Classical']
        X = df.drop(['name','label'],axis=1)
        X['beats'] = X['beats'].astype(float)
        y = df['label'].map(genre_map)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=20)
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        self.classifier = SVC(C=25,probability=True,decision_function_shape=('ovo'))
        self.classifier.fit(X_train_scaled,y_train)
        
        self.genres = [i for i in df.label.unique()]
        #print(self.genres)
    
    
    def get_song(self,songurl):
        '''Retrieves the song from youtube.
        input: songurl - the url of the song video.'''
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'mystery_song.%(ext)s'
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([songurl])
    
    
    def fit(self,songurl):
        '''Retrieves the specified song, compiles the features,
        and runs a prediciton model.
        input: songurl - the url of the song video to be classified.'''
        
        self.get_song(songurl)
        
        header = 'tempo beats chroma_stft rmse spec_cent spec_bw rolloff zcr'.split()
        header += ['mfcc_' + str(i) for i in range(1,12)]
        song_list = []

        songname = 'mystery_song.mp3'
        y_total, sr = librosa.load(songname, mono=True, duration=180, sr=None)
        for i, y in enumerate(np.array_split(y_total, 6)):
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rmse(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)

            seg_dict = {'tempo' : tempo,
                        'beats' : beats.shape[0],
                        'chroma_stft' : np.mean(chroma_stft),
                        'rmse' : np.mean(rmse),
                        'spec_cent' : np.mean(spec_cent),
                        'spec_bw' : np.mean(spec_bw), 
                        'rolloff' : np.mean(rolloff), 
                        'zcr' : np.mean(zcr)}    

            for j,e in enumerate(mfcc[1:]):
                seg_dict['mfcc_' + str(j+1)] = np.mean(e)

            song_list.append(seg_dict)
            
        song_df = pd.DataFrame(song_list,columns=header)
        song_df['beats'] = song_df['beats'].astype(float)
        self.song = song_df
        self.classify()
        #return song_df
    
    
    def classify(self):
        '''Runs a classifier on the current song.'''
        
        scaled_song = self.scaler.transform(self.song)
        probas = self.classifier.predict_proba(scaled_song)
        #print(probas)
        means = {g:round(np.mean(probas[:,i]),4) for i,g in enumerate(self.genres)}
        print("-----------------------")
        print("Probability of Genre:")
        print("-----------------------")
        for genre,mean in means.items():
            print(genre + " : " + str(mean*100) + "%")
        #return(means)