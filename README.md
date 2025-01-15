# Difficulty-Aware Sampling for Enhanced Replay Buffers

## TODO

- [x] comment most of the code
- [ ] comment code for squeezenet?
- [ ] delete irrelevant files
- [ ] combine analysis notebooks? or leave them as is
- [ ] remove useless imports
- [ ] t test

## Setup
After cloning this repository and navigating to its root folder, create a virtual environment and install all required packages:
```
conda create --name learn-deep python=3.10
conda activate learn-deep
pip install -r requirements.txt
```

## Preprocessing Training Data

To create the training data splits for two tasks, run the following command from the `src/` directory:
```
python -m preprocessing.cifar_10 --n 2 # For CIFAR-10
python -m preprocessing.cifar_100 --n 2 # For CIFAR-100
python -m preprocessing.ewf # For the EuropeWindFarm dataset
```

## Training

The training loop can be run after completing the preprocessing step. From the `src/` directory, execute the following command:

```
python -m models.cifar.train # for CIFAR-10 and CIFAR-100
python -m models.ewf.train # For the EuropeWindFarm dataset
```

The following flags can be provided:

- `--n`: The number of tasks trained on for the CIFAR datasets. This value has to match the preprocessing command that was previously run. Default: `2`
- `--classes`: The number of different classes in the training data for CIFAR. For CIFAR-10 and CIFAR-100, the value should be set to 10 or 100, respectively. Default: `10`
- `--epochs`: The number of epochs that should be run, per task. Default: `10`
- `--replay-buffer`: Replay buffer strategy, should either be empty, `uniform`, `weighted_mean`, or `mdfm`. Default: `None`.
- `--replay-weights`: A JSON dictionary indicating the weights of each of the metrics for the `weighted_mean` strategy. The weights are normalized automatically. Example: `{"vog": 1.0, "learning_speed": 1.0, "predictive_entropy": 1.0, "mutual_information": 1.0, "variation_ratio": 1.0, "mean_std_deviation": 1.0, "mc_variance": 0.0}'`. Default: `{}`.
- `--seed`: Seed for training (for reproducibility). In the paper, the three seeds for CIFAR are 69, 420 and 80085 and for EWF the five seeds are 31, 42, 69, 420 and 80085. Default: `42`.
- `--buffer-size`: The maximum size for the replay buffer, in percentage. Default: `10` (meaning 10% of the training data per task)
- `--cutoff-lower`: The lower cutoff threshold for the experiment. Default: `20` (meaning 20%)
- `--cutoff-upper`: The upper cutoff threshold for the experiment. Default: `20` (meaning 20%)
- `--wandb`: Binary flag included if one wants to use [WandB](https://wandb.ai/site) to track experiments.
- `--store-checkpoint`: If included, the model will run the experiment with the intended setup, but at the end of task 1 it will compute all metrics and save them, and also save the model as a checkpoint to run more experiments on the same seed with different metrics without having to retrain on task 1. Only valid for experiments with 2 tasks.
- `--use-checkpoint`: If included, the model will use the checkpoint saved in a previous call to run only training on task 2 (as the training on task 1 should be equivalent independent of the replay buffer strategy).

This is an example of the flags one can include in the call for classification (importantly set the "mc_variance" weight to 0)

```
python -m models.cifar.train --n 2 --classes 100 --epochs 50 --replay-buffer weighted_mean --replay_weights '{"vog": 1.0, "learning_speed": 1.0, "predictive_entropy": 1.0, "mutual_information": 1.0, "variation_ratio": 1.0, "mean_std_deviation": 1.0, "mc_variance": 0.0}' --wandb
```

This is an example for regression (importantly set the all mc-weights except for "mc_variance" to 0):

```
python -m models.ewf.train --epochs 50 --replay-buffer weighted_mean --replay_weights '{"vog": 0.1, "learning_speed": 0.1, "predictive_entropy": 0.0, "mutual_information": 0.0, "variation_ratio": 0.0, "mean_std_deviation": 0.0, "mc_variance": 0.1}' --wandb
```

For uniform sampling simply run:

```
python -m models.cifar.train --n 2 --classes 100 --epochs 50 --replay-buffer uniform --wandb
```
