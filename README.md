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

This is an example of the flags one can include in the call for classification (importantly set the "mc_variance" weight to 0)
```
python -m models.cifar.train --n 2 --classes 100 --epochs 50 --replay-buffer simple_sorted --replay_weights '{"vog": 1.0, "learning_speed": 1.0, "mc_entropy": 1.0, "mc_mutual_information": 1.0, "mc_variation_ratio": 1.0, "mc_mean_std": 1.0, "mc_variance": 0.0}' --wandb
```
This is an example for regression (importantly set the all mc-weights except for "mc_variance" to 0):
```
python -m models.ewf.train --epochs 50 --replay-buffer simple_sorted --replay_weights '{"vog": 0.1, "learning_speed": 0.1, "mc_entropy": 0.0, "mc_mutual_information": 0.0, "mc_variation_ratio": 0.0, "mc_mean_std": 0.0, "mc_variance": 0.1}' --wandb
```
The weights are normalised automatically

For uniform sampling simply run:
```
python -m models.cifar.train --n 2 --classes 100 --epochs 50 --replay-buffer uniform --wandb
```
