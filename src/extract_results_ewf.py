import pdb
import pandas as pd
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cutoff-lower", action="store", type=int, default=20, help="Percentage of lower cutoff")
parser.add_argument("--cutoff-upper", action="store", type=int, default=20, help="Percentage of upper cutoff")

args = parser.parse_args()

task_name = f'ewf_results'#_seed_{args.seed}_lower_{args.cutoff_lower}_upper_{args.cutoff_upper}'

with open(f"{task_name}.txt", "r") as f:
    logs = f.readlines()

results = []
values = []
for line in logs:
    if line.startswith("Running experiments for seed: "):
        pattern = r"seed:\s*(\d+),\s*lower:\s*(\d+),\s*upper:\s*(\d+)"

        match = re.search(pattern, line)
        
        seed = int(match.group(1))
        lower = int(match.group(2))
        upper = int(match.group(3))
    elif line.startswith("running with replay strategy: "):
        strategy = line.split("running with replay strategy: ")[1].strip()
        if strategy == "uniform":
            metric = "-"
    elif line.startswith("Running with replay weight: "):
        metric = line.split("Running with replay weight: ")[1].split(" ")[0]
    elif line.startswith("Task 1 test accuracy: [") or line.startswith("Task 2 test accuracy: [") or line.startswith("Task 3 test accuracy: [") or line.startswith("Task 4 test accuracy: ["):
        values += [float(elt) for elt in line.replace("]", "").split("[")[1].split(",")]

        print(values)
        if line.startswith("Task 4 test accuracy: ["):
            print(seed, lower, upper, strategy, metric, values)
            if len(values) == 2:
                values = ["-", values[0], "-", values[1]]
            results.append([seed, lower, upper, strategy, metric] + values)
            values = []

df = pd.DataFrame(results, columns=["Seed", "Lower Cutoff Value", "Upper Cutoff Value", "Strategy", "Metric", "Task 1 - 1", "Task 1 - 2", "Task 1 - 3", "Task 1 - 4", "Task 2 - 1", "Task 2 - 2", "Task 2 - 3", "Task 2 - 4", "Task 3 - 1", "Task 3 - 2", "Task 3 - 3", "Task 3 - 4", "Task 4 - 1", "Task 4 - 2", "Task 4 - 3", "Task 4 - 4"])
df.to_csv(f"{task_name}.csv", index=False)
print(df)