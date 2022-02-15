# Набор используемых библиотек все тот же.

import pandas as pd
from catboost import CatBoostClassifier, Pool

# Аналогично загружаем модель из файла.

model = CatBoostClassifier()
model.load_model('model.cbm')

# Аналогично читаем тестовые данные, создаем Pool.
# Однако теперь в нем не будет целевой переменной - ее мы предсказываем.

test = pd.read_csv('data/test.tsv', sep='\t')
test_data = Pool(test['libs'], text_features=[0])

# Наконец, делаем предсказание на тестовых данных.

final_prediction = model.predict(test_data)

# Записываем результаты в файл.

with open('results/prediction.txt', 'w') as f:
    print('prediction', file=f)
    for value in final_prediction:
        print(value, file=f)
