import torch 
from PIL import Image
import numpy as np
import wandb
import pdb


def compute_VoG(vog_data):
    gradient_matrices = vog_data["gradient_matrices"]
    _, epoch_labels = vog_data["input_data"]
    checkpoints = vog_data["checkpoints"]
    # calculate VoG
    gradient_matrices = torch.stack(gradient_matrices, axis=0)
    grad_means = torch.mean(gradient_matrices, axis=0)
    grad_variances = np.sqrt(1 / len(checkpoints)) * torch.sum(torch.pow(gradient_matrices - grad_means.unsqueeze(0), 2), axis=0)
    grad_variances = grad_variances.mean(axis=[1, 2]) # average over pixels    
    # normalise per class
    normalised_grad_variances = []
    for l in epoch_labels.unique():
        class_grad_variances = grad_variances[epoch_labels == l]
        normalized_class_values = (class_grad_variances - class_grad_variances.mean()).abs() / class_grad_variances.std()
        normalised_grad_variances.append(normalized_class_values)
    return normalised_grad_variances

def visualize_VoG(grad_variances, input_images, labels, num_imgs=10):
    input_images = input_images.to(labels.device)
    for i, l in enumerate(labels.unique()):
        _, top_idcs = torch.topk(grad_variances[i], k=num_imgs, largest=True, sorted=False)
        _, bottom_idcs = torch.topk(grad_variances[i], k=num_imgs, largest=False, sorted=False)
        top_imgs = input_images[labels == l][top_idcs, :, :, :]
        bottom_imgs = input_images[labels == l][bottom_idcs, :, :, :]
        top_images_list = []
        bottom_images_list = []        
        for j, (t_img, b_img) in enumerate(zip(top_imgs, bottom_imgs)):
            top_img_np = (255 * (t_img + 1) / 2.0).permute(1, 2, 0).byte().cpu().detach().numpy()
            bot_img_np = (255 * (b_img + 1) / 2.0).permute(1, 2, 0).byte().cpu().detach().numpy()
            top_images_list.append(wandb.Image(top_img_np, caption=f"Class {l+1} - Top {j+1}"))
            bottom_images_list.append(wandb.Image(bot_img_np, caption=f"Class {l+1} - Bottom {j+1}"))
        wandb.log({
            f"Top Images - Class {l+1}": top_images_list,
            f"Bottom Images - Class {l+1}": bottom_images_list
        })
    return
