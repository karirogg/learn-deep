import pdb
import pandas as pd


with open("results.txt", "r") as f:
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
df.to_csv("results.csv", index=False)
print(df)