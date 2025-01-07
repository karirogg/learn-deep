import pdb
import pickle
import torch


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

with open("checkpoints/metrics.pkl", "rb") as f:
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
base_metric = "learning_speed"
for metric, idcs in sorted_idcs.items():
    print(f"IoU of {metric} and {base_metric}:", round(compute_IoU(idcs, sorted_idcs[base_metric], cutoff_bottom, cutoff_top), 4))
    mean_DoI, num_idcs = compute_mean_DoI(idcs, sorted_idcs[base_metric], cutoff_bottom, cutoff_top)
    print(f"Mean distance of indices between {metric} and {base_metric} based on {num_idcs} values:", round(float(mean_DoI), 4))
    
# pdb.set_trace()