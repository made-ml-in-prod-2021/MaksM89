FROM python:3.6
RUN pip install gunicorn
COPY online_inference/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY ml_project/features/build_features.py /app/ml_project/features/build_features.py
COPY ml_project/enities/* /app/ml_project/enities/
COPY models/*.pkl /app/models/
COPY inference/server.py /app/server.py

WORKDIR /app

ENV MODELS_PATH="/app/models"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]