# Библиотеку scikit-learn будем использовать для подсчета требуемых метрик.
# Остальные библиотеки используются как и раньше.

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Читаем модель из ранее созданного файла.

model = CatBoostClassifier()
model.load_model('model.cbm')

# Загружаем данные для проверки метрик модели сначала в DataFrame, а затем сразу в Pool.
# Признак, целевая переменная и индикатор текстового признака остаются теми же.

valid = pd.read_csv('val.tsv', sep='\t')
valid_data = Pool(valid['libs'], valid['is_virus'], text_features=[0])

# Получем предсказание модели на новых валидационных данных.

valid_prediction = model.predict(valid_data)

# Теперь пишем в файл значения, которые выдает наша модель на требуемых метриках.

with open('validation.txt', 'w') as f:
    conf_matrix = confusion_matrix(valid['is_virus'], valid_prediction)
    print("True positive: ", conf_matrix[0][0], file=f)
    print("False positive", conf_matrix[0][1], file=f)
    print("False negative", conf_matrix[1][0], file=f)
    print("True negative", conf_matrix[1][1], file=f)
    print("Accuracy: ", accuracy_score(valid.is_virus, valid_prediction), file=f)
    print("Precision: ", precision_score(valid.is_virus, valid_prediction), file=f)
    print("Recall: ", recall_score(valid.is_virus, valid_prediction), file=f)
    print("F1: ", f1_score(valid.is_virus, valid_prediction), file=f)
