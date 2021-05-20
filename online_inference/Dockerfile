FROM python:3.6-slim
RUN pip install gunicorn==20.1.0
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY ml_project/features/build_features.py /app/ml_project/features/build_features.py
COPY ml_project/enities/* /app/ml_project/enities/
COPY models/*.pkl /app/models/
COPY online_inference/server.py /app/server.py

WORKDIR /app

ENV MODELS_PATH="/app/models"
# for debug uncomment and comment after. Then docker run --rm --name app -p 5000:5000 -it <imagename> bash
#ENV FLASK_APP=/app/server.py
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]