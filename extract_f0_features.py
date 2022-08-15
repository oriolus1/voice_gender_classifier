import numpy as np
import pandas as pd
import os
import librosa, librosa.display, librosa.filters


MAX_EXAMPLES_PER_READER = 3
PATH = 'E:/LibriTTS/train-clean-100/LibriTTS/train-clean-100'


def get_f0(filename: str):
    sound, sr = librosa.load(filename)
    f0, _, _ = librosa.pyin(sound, sr=sr, fmin=10, fmax=300, frame_length=1024)
    return [
        np.nanmean(f0),
        np.nanstd(f0),
        np.nanpercentile(f0, 25),
        np.nanmedian(f0),
        np.nanpercentile(f0, 75),
        np.nanpercentile(f0, 75) - np.nanpercentile(f0, 25)
    ], ['f0_mean', 'f0_std', 'f0_25p', 'f0_median', 'f0_75p', 'f0_iqr']


def collect_f0_features(path=PATH, max_examples_per_reader=None):
    dict_for_f0 = {}
    # counter contains number of processed examples per reader
    counter = {}
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                if max_examples_per_reader is None:
                    dict_for_f0[file], feature_names_f0 = get_f0(os.path.join(root, file))
                else:
                    reader_id = int(file.split("_")[0])
                    if counter.get(reader_id, 0) < max_examples_per_reader:
                        dict_for_f0[file], feature_names_f0 = get_f0(os.path.join(root, file))
                        counter[reader_id] = counter.get(reader_id, 0) + 1
                   
    df_with_f0 = pd.DataFrame.from_dict(dict_for_f0, orient='index')
    df_with_f0.columns = feature_names_f0
    df_with_f0 = pd.DataFrame.dropna(df_with_f0)
    df_with_f0 = df_with_f0.drop(df_with_f0[(df_with_f0.f0_mean > 300) | (df_with_f0.f0_mean < 10)].index)
    
    df_with_f0.to_csv('df_with_f0.csv')

    return df_with_f0

if __name__ == '__main__':
    collect_f0_features(path=PATH, max_examples_per_reader=MAX_EXAMPLES_PER_READER)