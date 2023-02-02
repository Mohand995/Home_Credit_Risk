FROM python:3.10

WORKDIR  /app

RUN pip install --upgrade pip

COPY  requirements.txt  .

RUN pip install -r requirements.txt

COPY  .  .

CMD ["python","Inference_api.py","--host","0,0,0,0","--port","5000"]

