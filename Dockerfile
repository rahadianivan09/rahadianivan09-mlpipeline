FROM tensorflow/serving:latest

COPY rahadianivan09-pipeline/serving_model /models/bank-deposit-model
ENV MODEL_NAME=bank-deposit-model

EXPOSE 8501