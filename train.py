# В качестве библиотеки для построения модели выбрал CatBoost, разработанную Яндексом.
# Выбор пал на нее в силу следующих соображений:
# 1) Библиотека реализует градиентный бустинг на решающих деревьях, что должно давать высокую точность.
# 2) Библиотека умеет работать с категориальными и текстовыми признаками - нужно лишь указать модели такие признаки.

import pandas as pd
from catboost import CatBoostClassifier, Pool

# Библиотека pandas использовалась в качестве основной для удобного хранения табличных данных.
# Прочитаем тренировочный файл. В процессе первичного ознакомления с данными я заметил очень необычную запись.
# Я изучил информацию в интернете, где узнал, что файл msvbvm60.dll действительно хранится по адресу c:\windows\system32.
# Исходя из этого, я принял решение просто заменить каждое вхождение этой записи на "msvbvm60.dll", списав аномалию на ошибку при получении данных через LIEF

train = pd.read_csv('data/train.tsv', sep='\t')
train = train.replace(
    'c:\\\\\\\\\\\\/\\/\\/\\/\\//\\\\\\\\\\\\\\\\\\\\\\\\\\\\//////\\\\/\\/\\/windows\\\\\\\\\\\\/\\/\\/\\/\\//\\\\\\\\\\\\\\\\\\\\\\\\\\\\//////\\\\/\\/\\/system32\\\\\\\\\\\\/\\/\\/\\/\\//\\\\\\\\\\\\\\\\\\\\\\\\\\\\//////\\\\\\/\\/\\/\\/\\\\/\\\\/\\/\\/\\/\\/\\\\/\\/\\/\\/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\/\\/\\/msvbvm60',
    'msvbvm60.dll')

# Далее определяем нашу модель. Поскольку наша задача является задачей классификации, используем CatBoostClassifier.
# Некоторые параметры модели я изменил, а конкретно некоторые моменты токенизации и составления словарей.
# Далее для каждого измененного параметра оставляю комментарий с пояснениями.

model = CatBoostClassifier(
    text_processing={
        "tokenizers": [{                             # Токенизировать будем названия библиотек, разделитель - запятая.
            "tokenizer_id": "Comma",                 # Значение по умолчанию - "Space".
            "separator_type": "ByDelimiter",
            "delimiter": ","                         # Значение по умолчанию - " ".
        }],

        "dictionaries": [{                           # В словари будем заносить все слова, независимо от частоты встречи.
            "dictionary_id": "BiGram",
            "max_dictionary_size": "-1",             # Значение по умолчанию - 50000, сделал неограниченным.
            "occurrence_lower_bound": "0",           # Значениб по умолчанию - 3, снял ограничение на частоту встречи.
            "gram_order": "2"
        }, {
            "dictionary_id": "Word",
            "max_dictionary_size": "-1",             # Значение по умолчанию - 50000, сделал неограниченным.
            "occurrence_lower_bound": "0",           # Значениб по умолчанию - 3, снял ограничение на частоту встречи.
            "gram_order": "1"
        }],

        "feature_processing": {                      # Откорректировал название применяемого токенизатора.
            "default": [{
                "dictionaries_names": ["BiGram", "Word"],
                "feature_calcers": ["BoW"],
                "tokenizers_names": ["Comma"]        # Значение по умолчанию - "Space".
            }, {
                "dictionaries_names": ["Word"],
                "feature_calcers": ["NaiveBayes"],
                "tokenizers_names": ["Comma"]        # Значение по умолчанию - "Space".
            }],
        }
    }, verbose=False)

# Создаем Pool для удобной и быстрой работы с данными для обучения модели.
# В параметрах передаем признак, целевую переменную, а также явно указываем на то, что наш признак - текстовый.

train_data = Pool(train['libs'], train['is_virus'], text_features=[0])

# Тренируем нашу модель, а затем сохраняем в файл .cbm - CatBoost Binary Format.

model.fit(train_data)
model.save_model('model.cbm')
