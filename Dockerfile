FROM python:3.7.4-slim-stretch

COPY . /usr/src/app
WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN pip install numpy
RUN pip install --ignore-installed -r requirements.txt

EXPOSE 5000
CMD [ "python" , "run_app.py"]