import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


model = CatBoostClassifier()
model.load_model('model.cbm')


valid = pd.read_csv('val.tsv', sep='\t')
valid_data = Pool(valid['libs'], valid['is_virus'], text_features=[0])


pred = model.predict(valid_data)


with open('validation.txt', 'w') as f:
    m = confusion_matrix(valid['is_virus'], pred)
    print("True positive: ", m[0][0], file=f)
    print("False positive", m[0][1], file=f)
    print("False negative", m[1][0], file=f)
    print("True negative", m[1][1], file=f)
    print("Accuracy: ", accuracy_score(valid.is_virus, pred), file=f)
    print("Precision: ", precision_score(valid.is_virus, pred), file=f)
    print("Recall: ", recall_score(valid.is_virus, pred), file=f)
    print("F1: ", f1_score(valid.is_virus, pred), file=f)
