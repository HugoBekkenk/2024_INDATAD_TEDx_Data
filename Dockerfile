FROM python:3.10-bookworm

RUN pip install psycopg2-binary
RUN pip install requests
RUN pip install youtube-transcript-api
RUN pip install joblib
RUN pip install numpy
RUN pip install scikit-learn


WORKDIR /app

ARG DB_NAME
ARG DB_USER
ARG DB_HOST
ARG DB_PASSWORD
ARG API_KEY

ENV DB_NAME=$DB_NAME
ENV DB_USER=$DB_USER
ENV DB_HOST=$DB_HOST
ENV DB_PASSWORD=$DB_PASSWORD
ENV API_KEY=$API_KEY

COPY main.py .
COPY kmeans_model.joblib .
COPY scaler.pkl .
COPY sentiment_model.joblib .
COPY vectorizer.joblib .

CMD ["sh","-c","python3 main.py"]