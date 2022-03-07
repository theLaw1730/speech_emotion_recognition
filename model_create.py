#-------------------------------------------------------------------------------------
#start imports
"""
to supress warning regarding version not matching up or something
From version 0.10 passing these as positional arguments will result in an error
(they were annoying)
"""

import warnings
warnings.filterwarnings('ignore')
#-------------------------------------------------------------------------------------
"""
to save the model
joblib > pickle because it is more efficient with objects that carry large numpy arrays
(that's what Google told me)
"""

import joblib
#-------------------------------------------------------------------------------------

import librosa
import soundfile
import os, glob
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

#end imports
#-------------------------------------------------------------------------------------

#dictionary of emotions, that the dataset comes with
emotions = {
    '01' : 'neutral',
    '02' : 'calm',
    '03' : 'happy',
    '04' : 'sad',
    '05' : 'angry',
    '06' : 'fearful',
    '07' : 'disgust',
    '08' : 'surprised'
}

#array of emotions, that we will work with/"observe"
emotions_to_observe = [
    'calm',
    'happy',
    'fearful',
    'disgust'
]

#extract features (chroma, mfcc, mel) from the the sound sample
def extract_features(file_name, chroma = True, mfcc = True, mel = True):
    with soundfile.SoundFile(file_name) as sound_file:
        results = np.array([])
        x = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        
        if chroma:
            stft = np.abs(librosa.stft(x))
            chroma = np.mean(librosa.feature.chroma_stft(S = stft, sr = sample_rate).T, axis = 0)
            results = np.hstack((results, chroma))
            
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y = x, sr = sample_rate, n_mfcc = 40).T, axis = 0)
            results = np.hstack((results, mfccs))
            
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(x, sr = sample_rate).T, axis = 0)
            results = np.hstack((results, mel))
    return results

#load dataset, extract featurs and split train and test sets
def load_data_and_split(test_size = 0.25):
    # x = features, y = emotions
    x, y = [], []
    for file in glob.glob("./dataset_smol/Actor_*/*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in emotions_to_observe:
            continue
        feature = extract_features(file, mfcc = True, chroma = True, mel = True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size = test_size)

#initializing model and saving (the good part, but you can't skip to it)
def model_maker(x_train, y_train):
    model = MLPClassifier(
                    alpha=0.01, 
                    batch_size=192, 
                    hidden_layer_sizes=(500,), 
                    learning_rate='adaptive', 
                    max_iter=500)
    model.fit(x_train, y_train)
    
    joblib.dump(model, 'model/smol_model_but_accurate.pkl')

def model_loader():
    return joblib.load('model/smol_model_but_accurate.pkl')
    
def model_accuracy(model, x_test, y_test):
    predictions = model.predict(x_test)
    return accuracy_score(y_test, predictions)

def main():
    x_train, x_test, y_train, y_test = load_data_and_split()
    model_maker(x_train, y_train)
    model = model_loader()
    
    print(model_accuracy(model, x_test, y_test))
    #current accuracy: 0.713542

if __name__ == '__main__':
    main()