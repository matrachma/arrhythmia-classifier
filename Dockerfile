FROM python:3.7.4

COPY . /usr/src/app
WORKDIR /usr/src/app

RUN pip install numpy
RUN pip install --ignore-installed -r requirements.txt


EXPOSE 5000
CMD [ "python" , "run_app.py"]