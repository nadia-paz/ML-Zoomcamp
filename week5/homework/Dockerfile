FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model1.bin", "dv.bin","./"]

EXPOSE 2912

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:2912", "predict:app"]