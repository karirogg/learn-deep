# Learn Deep Final Project

## Create Virtual Environment

```
python -m venv env
source env/bin/activate
```

## Running scripts

Example:

1. Navigate to `/src`:

```
cd src
```

2. Run scripts from `/src`:

```
python -m models.example
```

The file `/src/models/example.py` shows how to load the data in memory

## Preprocessing training data

```
python -m preprocessing.cifar_10 --n 2
```

## Running training loop

```
python -m models.training_loop --n 2
```
