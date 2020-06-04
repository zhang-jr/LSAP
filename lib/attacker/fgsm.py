import torch
import torch.nn as nn
import numpy as np

from attack import Attack

class FGSM(Attack):
    """
    FGSM attack in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 0.007)
    
    """
    def __init__(self, model, eps=0.007):
        super(FGSM, self).__init__("FGSM", model)
        self.eps = eps
    
    def forward(self, images, labels, target=False):
        images, labels = images.to(self.device), labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        
        images.requires_grad = True
        outputs = self.model(images)
        if target:
            random_targets = torch.randint(outputs.size(1), size=labels.shape, dtype=torch.int64, device=self.device)
            index = np.where(random_targets == labels)[0]
            for i in index:
                while random_targets[i].item() == labels[i].item():
                    random_targets[i] = torch.randint(outputs.size(1), size=(1, ), dtype=torch.int64, device=self.device)
            cost = loss(outputs, random_targets).to(self.device)

            grad = torch.autograd.grad(cost, images, 
                                    retain_graph=False, create_graph=False)[0]

            adv_images = images - self.eps*grad.sign()
            #adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            cost = loss(outputs, labels).to(self.device)
            #print("checkpoint: ", cost.item())
            
            grad = torch.autograd.grad(cost, images, 
                                    retain_graph=False, create_graph=False)[0]

            adv_images = images + self.eps*grad.sign()
            #adv_images = torch.clamp(adv_images, min=images.min().item(), max=images.max().item()).detach()

        return adv_images
    
    