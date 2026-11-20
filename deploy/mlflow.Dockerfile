FROM ghcr.io/mlflow/mlflow:v3.10.1@sha256:bb350de1ae29025e3f78d70a3443047ca863429cc577a28e050c7156ed8d388d
RUN python -m pip install --no-cache-dir psycopg2-binary==2.9.11
