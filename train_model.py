import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from create_dataframe_for_model import create_dataframe


PATH_TO_DATA = 'E:/LibriTTS/train-clean-100/LibriTTS/train-clean-100'
PATH_TO_MODEL = 'model.pkl'
TEST_RATIO = 0.25


def run_models_on_data(X_train, X_val, y_train, y_val):
    accuracy_list = []
    f1_list = []
    model_list = []

    models = [('LogReg', LogisticRegression(max_iter=10000)), 
              ('KNN', KNeighborsClassifier()), 
              ('RandomForest', RandomForestClassifier()),
              ('GradBoostClf', GradientBoostingClassifier()),
              ('SVM', SVC())]

    for name, model in models:
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        accuracy_list.append(accuracy_score(y_val, y_pred).round(3))
        f1_list.append(f1_score(y_val, y_pred).round(3))
        model_list.append(name)
        print('\n ******' + name, ': ')
        print('Accuracy score = ', accuracy_score(y_val, y_pred).round(3))
        print('F1 score = ', f1_score(y_val, y_pred).round(3))

    fig, ax = plt.subplots()
    ax.scatter(np.arange(0, len(model_list)), accuracy_list, color='green', label='accuracy')
    ax.scatter(np.arange(0, len(model_list)), f1_list, color='blue', label='f1 score')
    ax.set_xticks(np.arange(0, len(model_list)))
    ax.set_xticklabels(model_list)
    ax.legend()
    plt.show() 
    
    return model_list, accuracy_list, f1_list  

X_train, X_val, y_train, y_val = create_dataframe(path=PATH_TO_DATA, model_mode='training', test_ratio=TEST_RATIO)
print("\n Trying models on all data")
model_list, accuracy_list, f1_list = run_models_on_data(X_train, X_val, y_train, y_val)


# we want to analyze the impact on predictive power each part of extracted data: f0-related and MFCC-related
# so we will separate information now and run models on both parts separately
X_train_only_f0 = X_train[list(X_train.filter(regex='f0'))]
X_val_only_f0 = X_val[list(X_val.filter(regex='f0'))]
print("\n Trying models on f0 data only")
_, accuracy_list_f0, f1_list_f0 = run_models_on_data(X_train_only_f0, X_val_only_f0, y_train, y_val)

X_train_only_mfcc = X_train[list(X_train.filter(regex='mfcc'))]
X_val_only_mfcc = X_val[list(X_val.filter(regex='mfcc'))]
print("\n Trying models on MFCC data only")
_, accuracy_list_mfcc, f1_list_mfcc = run_models_on_data(X_train_only_mfcc, X_val_only_mfcc, y_train, y_val)


best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)
with open(PATH_TO_MODEL, 'wb') as f:
    pickle.dump(best_model, f)

