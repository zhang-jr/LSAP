import torch
import torch.nn as nn
import numpy as np
import math

from attack import Attack

class IPGDM(Attack):
    """
    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 4/255)
        alpha (float): alpha in the paper. (DEFALUT : 1/255)
        iters (int): max iterations. (DEFALUT : 0)
    
    .. note:: With 0 iters, iters will be automatically decided with the formula in the paper.
    """
    def __init__(self, model, eps=4, alpha=1, iters=0, num_classes=101):
        super(IPGDM, self).__init__("IPGDM", model)
        self.eps = eps
        self.alpha = alpha
        self.num_classes = num_classes
        if iters == 0 :
            #self.iters = int(min(eps*255 + 4, 1.25*eps*255))
            self.iters = int(min(eps + 4, 1.25*eps))
        else :
            self.iters = iters
        
    def forward(self, images, labels, target=True):
        """
        images: (B, 3*num_segments, 224, 224)
        labels: (B,)
        """
        images, labels = images.to(self.device), labels.to(self.device)
        perturbation = torch.zeros_like(images[0])
        perturbation.unsqueeze(dim=0)
        perturbation.requires_grad = True
        loss = nn.CrossEntropyLoss()

        if target:
            random_targets = torch.randint(self.num_classes, size=labels.shape, dtype=torch.int64, device=self.device)
            index = np.where(random_targets == labels)[0]
            for i in index:
                while random_targets[i].item() == labels[i].item():
                    random_targets[i] = torch.randint(self.num_classes, size=(1, ), dtype=torch.int64, device=self.device)
            labels = random_targets
        
        train_labels = labels.repeat(images.shape[0])

        for i in range(self.iters) :    
            adv_images = images + perturbation
            outputs = self.model(adv_images)

            if target:
                cost = loss(outputs, train_labels).to(self.device)
            
                grad = torch.autograd.grad(cost, perturbation,
                                       retain_graph=False, create_graph=False)[0]
            
                perturbation.data -= self.alpha*grad.sign()
                print('Norm: ', torch.norm(perturbation, p=math.inf).item())
                #adv_images = images  + perturbation
            else:
                
                cost = loss(outputs, train_labels).to(self.device)
            
                grad = torch.autograd.grad(cost, perturbation,
                                       retain_graph=False, create_graph=False)[0]
            
                perturbation.data += self.alpha*grad.sign()
                #adv_images = images + self.alpha*grad.sign()

            def clip(images, adv_images, min_value=[-104, -117, -128], max_value=[151, 138, 127]):
                adv_imgs = torch.zeros_like(images)
                for i in range(len(min_value)):
                    max_a = torch.clamp(images[:, i::3]-self.eps, min=min_value[i])
                    tmp = (adv_images[:, i::3] >= max_a).float()*adv_images[:, i::3] + (max_a > adv_images[:, i::3]).float()*max_a
                    min_c = (tmp > images[:, i::3] + self.eps).float()*(images[:, i::3] + self.eps) + (images[:, i::3] + self.eps >= tmp).float()*tmp
                    adv_imgs[:, i::3] = torch.clamp(min_c, max=max_value[i])
                return adv_imgs
            
            images = clip(images.detach(), adv_images.detach())

        predicted = torch.argmax(outputs)
        acc = (predicted == labels).sum().item()

        adv_images = images

        return adv_images, perturbation, torch.norm(perturbation, p=math.inf).item(), acc