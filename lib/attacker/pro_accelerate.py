import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from attack import Attack

class Proxi(Attack):
    """

    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon e-ball. (DEFALUT : 4/255)
        alpha (float): alpha update step. (DEFALUT : 1/255)
        iters (int): max iterations. (DEFALUT : 0)
    
    """
    def __init__(self, model, eps=4, alpha=1, iters=0, p=2):
        super(Proxi, self).__init__("Proxi", model)
        self.eps = eps
        self.alpha = alpha
        self.p = p
        self.optimizer = torch.optim.SGD
        if iters == 0 :
            #self.iters = int(min(eps*255 + 4, 1.25*eps*255))
            self.iters = int(min(eps + 4, 1.25*eps))
        else :
            self.iters = iters

    def generalize_project_descent(self, pertur, eps, p):
        # p=1, 2, math.inf
        # pertur1 = pertur * eps * F.normalize(perturbation, p, dim=-1)  #?

        if p == 2:
            pertur = pertur * min(1, eps / torch.norm(pertur))
        elif p == math.inf:
            pertur = pertur.sign() * torch.min(abs(pertur), eps*torch.ones_like(pertur))

        return pertur

        
    def forward(self, images, labels, target=True):
        images, labels = images.to(self.device), labels.to(self.device)
        perturbation = torch.zeros_like(images)
        perturbation.requires_grad = True
        later_perturbation, pre_perturbation = perturbation, perturbation
        #optimizer = self.optimizer(perturbation, lr=0.01)
        loss = nn.CrossEntropyLoss()
        
        for i in range(self.iters) :    
            inter_pertur = later_perturbation + (i/(i+3)) * (later_perturbation-pre_perturbation)
            pre_perturbation = later_perturbation
            adv_images = images + inter_pertur
            outputs = self.model(adv_images)

            if target:

                random_targets = torch.randint(outputs.size(1), size=labels.shape, dtype=torch.int64, device=self.device)
                index = np.where(random_targets == labels)[0]
                for i in index:
                    while random_targets[i].item() == labels[i].item():
                        random_targets[i] = torch.randint(outputs.size(1), size=(1, ), dtype=torch.int64, device=self.device)
                    
                cost = loss(outputs, random_targets).to(self.device)
                #cost2 = 
            
                grad = torch.autograd.grad(cost, inter_pertur,
                                       retain_graph=False, create_graph=False)[0]
            
                inter_pertur -= self.alpha*grad
                print('Before Project: ', torch.norm(inter_pertur, p=self.p).item())
                later_perturbation = self.generalize_project_descent(inter_pertur, self.eps, self.p)
                print('After project: ', torch.norm(later_perturbation, p=self.p).item())
            

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

        return adv_images, torch.norm(later_perturbation, p=self.p).item()