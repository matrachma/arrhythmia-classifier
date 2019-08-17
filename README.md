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

Move and rename your saved model to saved_models dir, so must be like:

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

Open http://127.0.0.1:5000/ 