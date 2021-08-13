import pandas as pd
from catboost import CatBoostClassifier, Pool


model = CatBoostClassifier()
model.load_model('model.cbm')


test = pd.read_csv('test.tsv', sep='\t')
test_data = Pool(test['libs'], text_features=[0])


final_prediction = model.predict(test_data)


with open('prediction.txt', 'w') as f:
    print('prediction', file=f)
    for value in final_prediction:
        print(value, file=f)
