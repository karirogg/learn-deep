import torch 
from PIL import Image
import numpy as np
import wandb
import pdb
from tqdm import tqdm


class VoG:
    
    def __init__(self, dataloader, epochs_per_task, num_checkpoints, task_id, is_classification):
        self.dataloader = dataloader
        self.checkpoint_idcs = np.linspace(0, epochs_per_task, num_checkpoints, endpoint=False, dtype=np.int32) # iterations at which to store gradients
        self.gradient_matrices = []
        self.task_id = task_id
        self.is_classification = is_classification
        self.result = None
        
    def update(self, model, epoch_idx):
        if self.task_id != 0 or not epoch_idx in self.checkpoint_idcs:
            return
        model.eval()
        gradients = []
        idx_list = []
        for inputs, labels, idcs in self.dataloader:
            inputs = inputs.clone()
            inputs.requires_grad = True
            ones = torch.ones(labels.shape).to(inputs.device)
            logits = model(inputs, task_id=0)
            selected_logits = logits[torch.arange(labels.shape[0]), labels]
            selected_logits.backward(ones)
            gradients.append(inputs.grad.data)
            idx_list.append(idcs)
        idx_list = torch.cat(idx_list)
        gradients = torch.cat(gradients, dim=0)
        sorted_gradients = torch.zeros_like(gradients)
        for i, idx in enumerate(idx_list):
            sorted_gradients[idx] = gradients[i]
        if self.is_classification:
            sorted_gradients = sorted_gradients.mean(axis=1) # average over channels
        self.gradient_matrices.append(sorted_gradients)
        return
    
    def finalise(self):
        gradient_matrices = torch.stack(self.gradient_matrices)
        grad_means = torch.mean(gradient_matrices, axis=0)
        grad_variances = np.sqrt(1 / len(self.checkpoint_idcs)) * torch.sum(torch.pow(gradient_matrices - grad_means.unsqueeze(0), 2), axis=0)
        if self.is_classification:
            grad_variances = grad_variances.mean(axis=[1, 2]) # average over pixels 
            # normalise per class
            # normalised_grad_variances = [(grad_variances - grad_variances.mean()) / grad_variances.std()]
            # normalised_grad_variances = []
            # for l in epoch_labels.unique():
            #     class_grad_variances = grad_variances[epoch_labels == l]
            #     normalized_class_values = (class_grad_variances - class_grad_variances.mean()).abs() / class_grad_variances.std()
            #     normalised_grad_variances.append(normalized_class_values)
        self.result = grad_variances
        return grad_variances
    
    def visualise(self, num_imgs=10):
        if not self.is_classification or self.result is None:
            return
        unsorted_images, unsorted_labels, idcs = map(torch.cat, zip(*[(img, labels, idx) for img, labels, idx in self.dataloader]))
        if len(unsorted_labels.unique()) == 5:
            mean, std = torch.Tensor([0.4914, 0.4822, 0.4465]), torch.Tensor([0.2470, 0.2435, 0.2616])
        elif len(unsorted_labels.unique()) == 50:
            mean, std = torch.Tensor([0.5071, 0.4865, 0.4409]), torch.Tensor([0.2673, 0.2564, 0.2762])
        else:
            raise NotImplementedError
        unsorted_images = unsorted_images * std.view(1, 3, 1, 1).to(unsorted_images.device) + mean.view(1, 3, 1, 1).to(unsorted_images.device)
        images, labels = torch.zeros_like(unsorted_images), torch.zeros_like(unsorted_labels)
        for i, idx in enumerate(idcs):
            images[idx] = unsorted_images[i]
            labels[idx] = unsorted_labels[i]
        for i, l in enumerate(labels.unique()):
            _, top_idcs = torch.topk(self.result[labels == l], k=num_imgs, largest=True, sorted=False)
            _, bottom_idcs = torch.topk(self.result[labels == l], k=num_imgs, largest=False, sorted=False)
            top_imgs = images[labels == l][top_idcs, :, :, :]
            bottom_imgs = images[labels == l][bottom_idcs, :, :, :]
            top_images_list = []
            bottom_images_list = []        
            for j, (t_img, b_img) in enumerate(zip(top_imgs, bottom_imgs)):                
                top_img_np = (255 * t_img).permute(1, 2, 0).byte().cpu().detach().numpy()
                bot_img_np = (255 * b_img).permute(1, 2, 0).byte().cpu().detach().numpy()
                top_images_list.append(wandb.Image(top_img_np, caption=f"Class {l+1} - Top {j+1}"))
                bottom_images_list.append(wandb.Image(bot_img_np, caption=f"Class {l+1} - Bottom {j+1}"))
            wandb.log({
                f"Top Images - Class {l+1}": top_images_list,
                f"Bottom Images - Class {l+1}": bottom_images_list
            })
        return
