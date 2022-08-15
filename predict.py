import pickle
import pandas as pd

from create_dataframe_for_model import create_dataframe

PATH_TO_DATA = '/data_for_inference/'
PATH_TO_MODEL = 'model.pkl'
PATH_TO_OUTPUT = '/data_for_inference/output.csv'


df = create_dataframe(PATH_TO_DATA, model_mode='inference')

with open(PATH_TO_MODEL, 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(df)
y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv(PATH_TO_OUTPUT)