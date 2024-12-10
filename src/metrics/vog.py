import torch 
from PIL import Image
import numpy as np

def compute_VoG(grad_matrices, epoch_labels, checkpoints):
    # calculate VoG
    grad_matrices = torch.stack(grad_matrices, axis=0)
    grad_means = torch.mean(grad_matrices, axis=0)
    grad_variances = np.sqrt(1 / len(checkpoints)) * torch.sum(torch.pow(grad_matrices - grad_means.unsqueeze(0), 2), axis=0)
    grad_variances = grad_variances.mean(axis=[1, 2]) # average over pixels    
    # normalise per class
    normalised_grad_variances = []
    for l in epoch_labels.unique():
        class_grad_variances = grad_variances[epoch_labels == l]
        normalized_class_values = (class_grad_variances - class_grad_variances.mean()).abs() / class_grad_variances.std()
        normalised_grad_variances.append(normalized_class_values)
    return normalised_grad_variances

def visualize_VoG(grad_variances, input_images, labels, num_imgs=10):
    for i, l in enumerate(labels.unique()):
        _, top_idcs = torch.topk(grad_variances[i], k=num_imgs, largest=True, sorted=False)
        _, bottom_idcs = torch.topk(grad_variances[i], k=num_imgs, largest=False, sorted=False)
        top_imgs = input_images[labels == l][top_idcs, :, :, :]
        bottom_imgs = input_images[labels == l][bottom_idcs, :, :, :]
        for j, (t_img, b_img) in enumerate(zip(top_imgs, bottom_imgs)):
            Image.fromarray(t_img.permute(1, 2, 0).byte().cpu().detach().numpy()).save(f"../visus/vog/top_picks/class_{l}_pick_{j}.png")
            Image.fromarray(b_img.permute(1, 2, 0).byte().cpu().detach().numpy()).save(f"../visus/vog/bottom_picks/class_{l}_pick_{j}.png")
    return
