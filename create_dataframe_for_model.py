import pandas as pd
import os
import random

from extract_f0_features import collect_f0_features
from extract_mfcc_features import collect_mfcc_features

PATH = 'C:/Users/zyvyhome/Downloads/dev-clean/LibriTTS/dev-clean'
# TEST_RATIO = 0.33
RANDOM_SEED = 57

def is_male(df_readers: pd.DataFrame, reader_id: int):
    return 1 if df_readers.loc[df_readers.index == reader_id]['READER'].iloc[0] == 'M' else 0


def train_test_split_for_gender_lists(readers_list, number_of_examples_per_reader_dict, examples_number, test_ratio):
    test_readers_list = []
    train_readers_list = []
    test_counter = 0
    for reader in readers_list:
        if test_counter < test_ratio * examples_number:
            test_readers_list.append(reader)
            test_counter += number_of_examples_per_reader_dict[reader]
        else:
            train_readers_list.append(reader)
    return train_readers_list, test_readers_list    

    
def create_dataframe(path, model_mode='inference', test_ratio=None):    

    if model_mode == 'inference':
        # extract two types of features and combine them
        df_with_f0 = collect_f0_features(path)
        df_with_mfcc = collect_mfcc_features(path)
        df = df_with_f0.join(df_with_mfcc)
        
        return df


    # if we are to train/validate model, let's perform a train/validation split
    # and download pre-extracted features from already prepared files
    else:
        number_of_examples_per_reader_dict = {}
    
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".wav"):
                    reader_id = int(file.split("_")[0])
                    number_of_examples_per_reader_dict[reader_id] = number_of_examples_per_reader_dict.get(reader_id, 0) + 1
            
        # this dataframe is read from database and is not changed
        df_readers = pd.read_csv('speakers.tsv', sep='\t')
    
        # for our purposes we design 'readers_catalog' dataframe with columns:  	'number_of_examples' 	'gender' 	'reader_id' 	'train_or_test'
        readers_catalog = pd.DataFrame.from_dict(number_of_examples_per_reader_dict, orient='index')
        readers_catalog.columns = ['number_of_examples']
        readers_catalog['gender'] = readers_catalog.index
        readers_catalog['gender'] = readers_catalog['gender'].apply(lambda x: is_male(df_readers, x))
        readers_catalog['reader_id'] = readers_catalog.index
    
        # let's calculate how many male and female examples we've got    
        male_examples_number = int(readers_catalog.loc[readers_catalog['gender'] == 1, ['number_of_examples']].sum())
        female_examples_number = int(readers_catalog.loc[readers_catalog['gender'] == 0, ['number_of_examples']].sum())
    
        print("There are {} male and {} female examples in our data".format(male_examples_number, female_examples_number))
    
        male_list = readers_catalog.loc[readers_catalog['gender'] == 1, 'reader_id'].tolist()
        female_list = readers_catalog.loc[readers_catalog['gender'] == 0, 'reader_id'].tolist()

        print("There are {} male and {} female readers in our data".format(len(male_list), len(female_list)))
    
        # we can fix random seed here to fix train-test split for a given test ratio
        # random.Random(RANDOM_SEED).shuffle(male_list)
        # random.Random(RANDOM_SEED).shuffle(female_list)
    
        # now we compose test lists of male and female speakers by adding random readers one-by-one until there are enough wav files in the test part (and the rest is train)
        # we treat male and female lists separately so as to obtain good stratification in train-test split
        train_male_list, test_male_list = train_test_split_for_gender_lists(male_list, number_of_examples_per_reader_dict, male_examples_number, test_ratio)
        train_female_list, test_female_list = train_test_split_for_gender_lists(female_list, number_of_examples_per_reader_dict, female_examples_number, test_ratio)
    
        train_list = train_male_list + train_female_list
        test_list = test_male_list + test_female_list
    
        readers_catalog['train_or_test'] = readers_catalog.index
        readers_catalog['train_or_test'] = readers_catalog['train_or_test'].apply(lambda x: 'train' if x in train_list else 'test')

        # now we download pre-extracted features and combine all information together
        df_with_f0 = pd.read_csv('df_with_f0.csv', index_col=0)
        df_with_mfcc = pd.read_csv('df_with_mfcc.csv', index_col=0)
        df = df_with_f0.join(df_with_mfcc)
        
        df['reader_id'] = df.index
        df['reader_id'] = df['reader_id'].apply(lambda x: int(x.split("_")[0]))
        
        df = readers_catalog.merge(df)
    
        # get X_train and y_train from dataframe
        X_train = df[df['train_or_test'] == 'train'].copy()
        X_train = X_train.drop(['number_of_examples', 'gender', 'reader_id', 'train_or_test'], axis=1)
        y_train = df[df['train_or_test'] == 'train']['gender'].copy()
    
        # get X_test and y_test from dataframe
        X_test = df[df['train_or_test'] == 'test'].copy()
        X_test = X_test.drop(['number_of_examples', 'gender', 'reader_id', 'train_or_test'], axis=1)
        y_test = df[df['train_or_test'] == 'test']['gender'].copy()
    
        return X_train, X_test, y_train, y_test
