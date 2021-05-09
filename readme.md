ml_example
==============================

github ***MaksM89***

discord **Maxibon#1812**

Example of ml project

# Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

# Data

[Dataset]("https://raw.githubusercontent.com/rashida048/Datasets/master/Heart.csv") downloads automaticaly.

# Usage (testing on Windows):

## Train model:
~~~
set pythonpath=%pythonpath%;%cd%
python ml_project/train_pipeline.py configs/train_config.yml
~~~
## Get prediction from model.

You need to train your model, serialized it in `/model` dir, then get name without `.pkl`. Pass model name and data in json format:
~~~
set pythonpath=%pythonpath%;%cd%
python ml_project/predict.py model {\"Age\":{\"1\":63,\"2\":67},\"Sex\":{\"1\":1,\"2\":1},\"ChestPain\":{\"1\":\"typical\",\"2\":\"asymptomatic\"},\"RestBP\":{\"1\":145,\"2\":160},\"Chol\":{\"1\":233,\"2\":286},\"Fbs\":{\"1\":1,\"2\":0},\"RestECG\":{\"1\":2,\"2\":2},\"MaxHR\":{\"1\":150,\"2\":108},\"ExAng\":{\"1\":0,\"2\":1},\"Oldpeak\":{\"1\":2.3,\"2\":1.5},\"Slope\":{\"1\":3,\"2\":2},\"Ca\":{\"1\":0.0,\"2\":3.0},\"Thal\":{\"1\":\"fixed\",\"2\":\"normal\"},\"AHD\":{\"1\":\"No\",\"2\":\"Yes\"}}
~~~
Also you may run Flask application and send requests by curl:
~~~
set flask_app=ml_project/predict.py
flask run
~~~
and in another cmd window:
~~~
curl '127.0.0.1:5000/query?modelname=model&data=\{"Age":\{"1":63,"2":67\},"Sex":\{"1":1,"2":1\},"ChestPain":\{"1":"typical","2":"asymptomatic"\},"RestBP":\{"1":145,"2":160\},"Chol":\{"1":233,"2":286\},"Fbs":\{"1":1,"2":0\},"RestECG":\{"1":2,"2":2\},"MaxHR":\{"1":150,"2":108\},"ExAng":\{"1":0,"2":1\},"Oldpeak":\{"1":2.3,"2":1.5\},"Slope":\{"1":3,"2":2\},"Ca":\{"1":0.0,"2":3.0\},"Thal":\{"1":"fixed","2":"normal"\},"AHD":\{"1":"No","2":"Yes"\}\}'
~~~
# Run test:
~~~
pytest tests/
~~~

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── ml_example                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- code to download or generate data
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │
    │   ├── models         <- code to train models and then use trained models to make
==============================

В данном проекте не сделано:

11) Используется hydra  (https://hydra.cc/docs/intro/) (3 балла - доп баллы)
12) Настроен CI(прогон тестов, линтера) на основе github actions  (3 балла - доп баллы (будем проходить дальше в курсе, но если есть желание поразбираться - welcome)
Т.е. -6 баллов. 

Взял за основу базовое решение, разобрался, кое-что поменял, конечно, не существенно. Положил в папку ноутбук для вида, нужно туда добавить ML. Ну и модель самая простая. Разобрался с импортами, marshmellow-dataclass. Буду рад любым замечаниям.


