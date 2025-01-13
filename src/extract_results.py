import pdb
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cutoff-lower", action="store", type=int, default=20, help="Percentage of lower cutoff")
parser.add_argument("--cutoff-upper", action="store", type=int, default=20, help="Percentage of upper cutoff")

args = parser.parse_args()

task_name = f'results_seed_{args.seed}_lower_{args.cutoff_lower}_upper_{args.cutoff_upper}'

with open(f"{task_name}.txt", "r") as f:
    logs = f.readlines()

results = []
values = []
for line in logs:
    if line.startswith("Running experiments for seed: "):
        seed = int(line.split("Running experiments for seed: ")[1])
    elif line.startswith("running with replay strategy: "):
        strategy = line.split("running with replay strategy: ")[1].strip()
        if strategy == "uniform":
            metric = "-"
    elif line.startswith("Running with replay weight: "):
        metric = line.split("Running with replay weight: ")[1].split(" ")[0]
    elif line.startswith("Task 1 test accuracy: [") or line.startswith("Task 2 test accuracy: ["):
        values += [float(elt) for elt in line.replace("]", "").split("[")[1].split(",")]
        if line.startswith("Task 2 test accuracy: ["):
            print(seed, strategy, metric, values)
            if len(values) == 2:
                values = ["-", values[0], "-", values[1]]
            results.append([seed, strategy, metric] + values)
            values = []

df = pd.DataFrame(results, columns=["Seed", "Strategy", "Metric", "Task 1 Initial Accuracy", "Task 1 Final Accuracy", "Task 2 Initial Accuracy", "Task 2 Final Accuracy"])
df.to_csv(f"{task_name}.csv", index=False)
print(df)