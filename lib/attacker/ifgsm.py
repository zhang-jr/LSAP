import torch
import torch.nn as nn
import numpy as np

from attack import Attack

class IFGSM(Attack):
    """
    I-FGSM attack in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 4/255)
        alpha (float): alpha in the paper. (DEFALUT : 1/255)
        iters (int): max iterations. (DEFALUT : 0)
    
    .. note:: With 0 iters, iters will be automatically decided with the formula in the paper.
    """
    def __init__(self, model, eps=4, alpha=1, iters=0):
        super(IFGSM, self).__init__("IFGSM", model)
        self.eps = eps
        self.alpha = alpha
        if iters == 0 :
            #self.iters = int(min(eps*255 + 4, 1.25*eps*255))
            self.iters = int(min(eps + 4, 1.25*eps))
        else :
            self.iters = iters
        
    def forward(self, images, labels, target=False):
        images, labels = images.to(self.device), labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        
        for i in range(self.iters) :    
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
            
                adv_images = images - self.alpha*grad.sign()
            else:
                cost = loss(outputs, labels).to(self.device)
            
                grad = torch.autograd.grad(cost, images,
                                       retain_graph=False, create_graph=False)[0]
            
                adv_images = images + self.alpha*grad.sign()
            

            def clip(images, adv_images, min_value=[-104, -117, -128], max_value=[151, 138, 127]):
                adv_imgs = torch.zeros_like(images)
                for i in range(len(min_value)):
                    max_a = torch.clamp(images[:, i::3]-self.eps, min=min_value[i])
                    tmp = (adv_images[:, i::3] >= max_a).float()*adv_images[:, i::3] + (max_a > adv_images[:, i::3]).float()*max_a
                    min_c = (tmp > images[:, i::3] + self.eps).float()*(images[:, i::3] + self.eps) + (images[:, i::3] + self.eps >= tmp).float()*tmp
                    adv_imgs[:, i::3] = torch.clamp(min_c, max=max_value[i])
                return adv_imgs
            
            images = clip(images.detach(), adv_images.detach())

        adv_images = images

        return adv_images