Общий отчет лежит в summary.pdf

##### Чтобы запустить train/validate: 

`python train_model.py`

##### Чтобы запустить inference:

`python predict.py`

Данные для inference нужно класть в /data_for_inference/, туда же пишется прогноз

Предобработанные фичи лежат в файлах df_with_f0.csv и df_with_mfcc.csv.

##### Если нужно пересчитать признаки:
`python extract_f0_features.py` и `python extract_mfcc_features.py`

(путь к данным задается константой PATH внутри этих файлов)
