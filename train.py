import pandas as pd
from catboost import CatBoostClassifier, Pool

train = pd.read_csv('train.tsv', sep='\t')
train = train.replace(
    'c:\\\\\\\\\\\\/\\/\\/\\/\\//\\\\\\\\\\\\\\\\\\\\\\\\\\\\//////\\\\/\\/\\/windows\\\\\\\\\\\\/\\/\\/\\/\\//\\\\\\\\\\\\\\\\\\\\\\\\\\\\//////\\\\/\\/\\/system32\\\\\\\\\\\\/\\/\\/\\/\\//\\\\\\\\\\\\\\\\\\\\\\\\\\\\//////\\\\\\/\\/\\/\\/\\\\/\\\\/\\/\\/\\/\\/\\\\/\\/\\/\\/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\/\\/\\/msvbvm60',
    'msvbvm60.dll')


model = CatBoostClassifier(dictionaries=[{
    "dictionary_id": "BiGram",
    "max_dictionary_size": "50000",
    "occurrence_lower_bound": "0",
    "gram_order": "2"
}, {
    "dictionary_id": "Word",
    "max_dictionary_size": "50000",
    "occurrence_lower_bound": "0",
    "gram_order": "1"
}], verbose=False)
train_data = Pool(train['libs'], train['is_virus'], text_features=[0])


model.fit(train_data)


model.save_model('model.cbm')
