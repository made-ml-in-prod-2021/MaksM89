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

## Local service (only for testing and debugging)

~~~
set FLASK_APP=online_inference/server.py
flask run
~~~

## Get docker image

### Build docker image

You need to train your model before, serialized it in `/model` dir. Docker image will add this files to container:
~~~
docker build -t webapp:v1 -f online_inference/dockerfile .
~~~

### Load image from github

You can load image form github with my model (1.2 GB):
~~~
docker pull docker.pkg.github.com/made-ml-in-prod-2021/maksm89/webapp:v1
docker tag docker.pkg.github.com/made-ml-in-prod-2021/maksm89/webapp:v1 webapp:v1
docker rmi docker.pkg.github.com/made-ml-in-prod-2021/maksm89/webapp:v1
~~~

## Than can run inference:
~~~
docker run -rm -p 5000:5000 --name webapp webapp:v1
~~~

and send requests by command:
~~~
python inference/requester.py <model_name> <path/to/csv>
python inference/requester.py model data/heart.csv
python inference/requester.py model data/request.csv
python inference/requester.py model data/request_small.csv
~~~

IP adress, used in requester.py - 192.168.99.100. Maybe you must change this yourself.

If model not found, server return error 501.

If don't have enough coumns, server return 503.

How to use: http://\<ip\>:5000/

You can see avaiable models in http://\<ip\s>:5000/models

# Run test:
~~~
pytest ml_project/tests/
~~~

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── online_inference   <- Code for start rest-service.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── ml_example                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- code to download or generate data, the original, immutable data dump.
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │
    │   ├── models         <- code to train models and then use trained models to make, trained and serialized models, 
                                model predictions, or model summaries
==============================

## hw1
В данном проекте не сделано:

11) Используется hydra  (https://hydra.cc/docs/intro/) (3 балла - доп баллы)
12) Настроен CI(прогон тестов, линтера) на основе github actions  (3 балла - доп баллы (будем проходить дальше в курсе, но если есть желание поразбираться - welcome)
Т.е. -6 баллов. 

Взял за основу базовое решение, разобрался, кое-что поменял, конечно, не существенно. Положил в папку ноутбук для вида, нужно туда добавить ML. Ну и модель самая простая. Разобрался с импортами, marshmellow-dataclass. Буду рад любым замечаниям.

## hw2

На мой взгляд, сделано всё, кроме оптимизации. Очень большой образ. Нужно взять за основу что-нибудь полегче.
Образ опубликован на гитхабе в том же проекте.

**Хочу баллов** = 15



