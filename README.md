# Arryhthmia Classifier

Simple UI for Arryhthmia Classifier

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

Use python 3.7.4

```
$ python
Python 3.7.4 (default, Aug 11 2019, 21:48:06) 
[Clang 10.0.1 (clang-1001.0.46.4)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

Move and rename your saved model (*.h5 file) to saved_models dir, so must be like:

```
arrhythmia-classifier
    |-saved_models
        |- .placeholder
        |- adabound.h5
        |- adadelta.h5
        |- adagrad.h5
        |- adam.h5
        |- amsbound.h5
        |- sgd.h5
    |-static
    |-temp
    |-.....
```

## Installing

Create python virtual environment or just simply install required package.

```
pip install -r requirements.txt
```

## Running

```
python -m run_app.py
```

Open http://localhost:5000 

## Docker Installation

### Build and run an image for the app 
```shell
$ cd arrhythmia-classifier
$ docker build -t arrhythmia-image .
$ docker run --name arrhythmia-app -d -p 5000:5000 arrhythmia-image 
```

### Pull an built-image from Docker hub
Will be available soo.

### Running
Open http://localhost:5000 after waiting for a while to install in the container.

![form-0](https://user-images.githubusercontent.com/8687198/63220163-33da7000-c1ac-11e9-87ee-0a8c6d1eba9b.png)

![form-1](https://user-images.githubusercontent.com/8687198/63220164-34730680-c1ac-11e9-82b7-40d8a626a824.png)


![result-0](https://user-images.githubusercontent.com/8687198/63220165-34730680-c1ac-11e9-923d-b4b11a4a2b72.png)

![result-1](https://user-images.githubusercontent.com/8687198/63220166-34730680-c1ac-11e9-8ac2-5170af7e4e53.png)
