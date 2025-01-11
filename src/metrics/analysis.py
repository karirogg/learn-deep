import pdb
import pickle
import torch
from tabulate import tabulate


def get_expected_IoU(n, u):
    expected_intersection = pow(n,2) / u
    expected_union = 2*n - pow(n,2) / u
    return expected_intersection / expected_union

def get_expected_mean_DoI(n):
    return n/3.0

def compute_IoU(idcs1, idcs2, cutoff_bottom, cutoff_top):
    cutoff_bottom = int(len(idcs1) * cutoff_bottom)
    cutoff_top = int(len(idcs1) * (1-cutoff_top))
    idcs1 = set(idcs1[cutoff_bottom:cutoff_top].int().tolist())
    idcs2 = set(idcs2[cutoff_bottom:cutoff_top].int().tolist())
    intersection = len(idcs1 & idcs2)
    union = len(idcs1 | idcs2)
    return intersection / union

def compute_mean_DoI(idcs1, idcs2, cutoff_bottom, cutoff_top):
    cutoff_bottom = int(len(idcs1) * cutoff_bottom)
    cutoff_top = int(len(idcs1) * (1-cutoff_top))
    idcs1 = idcs1[cutoff_bottom:cutoff_top]
    idcs2 = idcs2[cutoff_bottom:cutoff_top]
    common_elements = set(idcs1.int().tolist()) & set(idcs2.int().tolist())
    if len(common_elements) == 0:
        return float("nan")
    distances = []
    for elt in common_elements:
        pos1 = (idcs1 == elt).nonzero()[0]
        pos2 = (idcs2 == elt).nonzero()[0]
        distances.append(abs(pos1-pos2))
    distances = torch.hstack(distances).float()
    return distances.mean(), len(common_elements)

with open("checkpoints/training_metrics.pkl", "rb") as f:
    metrics = pickle.load(f)
    
sorted_idcs = {}
for metric, values in metrics.items():
    if metric != "learning_speed":
        sorted_idcs[metric] = torch.Tensor(sorted(torch.arange(len(values)), key=lambda i : values[i]))
    else:
        sorted_idcs[metric] = torch.Tensor(sorted(torch.arange(len(values)), key=lambda i : values[i], reverse=True))
       
cutoff_bottom = 0.2
cutoff_top = 0.2
u = len(values)
n = int(len(values) * (1-cutoff_bottom-cutoff_top))
print("Expected IoU:", round(get_expected_IoU(n=n, u=u), 4))
print("Expected Mean DoI", round(get_expected_mean_DoI(n=n), 4))
print()
# base_metric = "learning_speed"
IoU_rows = []
DoI_rows = []
metric_names, idcs = zip(*sorted_idcs.items())
metric_names = list(metric_names)
idcs = list(idcs)
for metric1, idcs1 in zip(metric_names, idcs):
    IoU_row = [metric1]
    DoI_row = [metric1]
    for metric2, idcs2 in zip(metric_names, idcs):
        IoU = round(compute_IoU(idcs1, idcs2, cutoff_bottom, cutoff_top), 4)
        mean_DoI, num_idcs = compute_mean_DoI(idcs1, idcs2, cutoff_bottom, cutoff_top)
        mean_DoI = round(float(mean_DoI), 4)
        IoU_row.append(IoU)
        DoI_row.append(mean_DoI)
    IoU_rows.append(IoU_row)
    DoI_rows.append(DoI_row)

print("Intersection over Union")
print(tabulate(IoU_rows, headers=metric_names, tablefmt="grid"))
print("Mean Distance of Indices")
print(tabulate(DoI_rows, headers=metric_names, tablefmt="grid"))
    
        
        # print(f"IoU of {metric} and {base_metric}:", round(compute_IoU(idcs, sorted_idcs[base_metric], cutoff_bottom, cutoff_top), 4))
        # mean_DoI, num_idcs = compute_mean_DoI(idcs, sorted_idcs[base_metric], cutoff_bottom, cutoff_top)
        # print(f"Mean distance of indices between {metric} and {base_metric} based on {num_idcs} values:", round(float(mean_DoI), 4))
    
# pdb.set_trace()