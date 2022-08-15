import numpy as np
import pandas as pd
import os
import librosa, librosa.display, librosa.filters

MAX_EXAMPLES_PER_READER = 3
NUMBER_OF_MFCC = 13
PATH = 'E:/LibriTTS/train-clean-100/LibriTTS/train-clean-100'


# this function extracts 13 mfcc coefficients from a single wav file 
# and returns their means and stds, as well as stds of delta and delta_order_2
def get_features(filename: str):
    sound, sr = librosa.load(filename)

    mfccs = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=NUMBER_OF_MFCC)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

    nums = np.arange(0, NUMBER_OF_MFCC)
    feature_names = ['mean_mfcc_' + str(num) for num in nums] + ['std_mfcc_' + str(num) for num in nums] 
    feature_names = feature_names + ['std_mfccs_delta_' + str(num) for num in nums] + ['std_mfccs_delta2_' + str(num) for num in nums]  

   
    return np.concatenate((
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1), np.std(mfccs_delta, axis=1), np.std(mfccs_delta2, axis=1)
    )), feature_names


def collect_mfcc_features(path=PATH, max_examples_per_reader=None):    
    dict_for_mfccs = {}
    # counter contains number of processed examples per reader
    # it is needed only if we want to forcibly limit number of examples per reader
    counter = {}
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                if max_examples_per_reader is None:
                    dict_for_mfccs[file], feature_names = get_features(os.path.join(root, file))
                else:
                    reader_id = int(file.split("_")[0])
                    if counter.get(reader_id, 0) < max_examples_per_reader:
                        dict_for_mfccs[file], feature_names = get_features(os.path.join(root, file))
                        counter[reader_id] = counter.get(reader_id, 0) + 1
    
    df_with_mfccs = pd.DataFrame.from_dict(dict_for_mfccs, orient='index')
    df_with_mfccs.columns = feature_names
    df_with_mfccs.to_csv('df_with_mfcc.csv')
    
    return df_with_mfccs


if __name__ == '__main__':
    collect_mfcc_features(PATH, MAX_EXAMPLES_PER_READER)