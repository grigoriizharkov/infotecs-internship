# Тестовое задание на стажировку в ИнфоТеКС

Задача состояла в следующем: разработать систему машинного обучения, которая по списку статически импортируемых библиотек exe файла предсказывает, является ли этот файл зловредным. 

Были предоставлены файлы с исходными данными (папка data): тренировочная, валидационная и тестовая выборка. В соответствие этим 3 датасетам были созданы .py файлы: train, val и predict. В папке results представлены результаты работы модели на валидацинной выборке, а также предсказания по тестовой выборке. Файл requirments.txt представляет собой список всех использованных библиотек и их версии. Файл model.cbm используется для сохранения модели. 


Мною была выбрана библиотека CatBoost из-за простоты обработки текстовых признаков (других в принципе не было). Модель показала на удивление хороший результат на валидационной выборке. Благодаря этому тестовому заданию мне удалось попасть на собеседование, однако в итоге на стажирвоку я не попал.
