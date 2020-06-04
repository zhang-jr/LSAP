import numpy as np
import torch
import torch.nn as nn
from attack import Attack


class DeepFool(Attack):

    def __init__(self, model, eps=0.150, max_iter=100):
        super(DeepFool, self).__init__('DeepFool', model)
        self.eps = eps
        self.max_iter = max_iter
        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, images):
        # input
        images = images.to(self.device)
        images.requires_grad = True

        # loss function
        criterion = nn.CrossEntropyLoss()

        #images.requires_grad = True
        outputs = self.model(images)
        #loss_1 = criterion(outputs, labels)

        pred_all_labels = torch.sort(-outputs)[1]
        pred_labels = pred_all_labels[:, 0]

        #pred_labels = torch.argmax(outputs, dim=1)
        w = torch.zeros_like(images)
        r_tot = torch.zeros_like(images)

        loop_i = 0
        labels_flag = pred_labels

        while labels_flag == pred_labels and loop_i < self.max_iter:

            pertur = torch.Tensor([float('Inf')]).to(self.device)
            loss_1 = criterion(outputs, pred_labels)   # loss_1 = 0?
            grad_1 = torch.autograd.grad(loss_1, images, retain_graph=True)[0]
            # grad_1 = torch.autograd.grad(outputs.gather(1, pred_labels.unsqueeze(dim=1)).mean(), images)[0] 

            for i in range(1, outputs.size(1)):
                
                loss_i = criterion(outputs, pred_all_labels[:, i])
                grad_i = torch.autograd.grad(loss_i, images, retain_graph=True)[0]
                # grad_i = torch.autograd.grad(outputs.gather(1, pred_all_labels[:, i).unsqueeze(dim=1)].mean(), images)[0]

                # set new w_i and output_i
                w_i = grad_i - grad_1
                output_i = outputs.gather(1, pred_all_labels[:, i].unsqueeze(1)) - outputs.gather(1, pred_labels.unsqueeze(1))

                #pertur_i = abs(output_i) / w_i.norm(2, -1).norm(2, -1). norm(2, -1)
                pertur_i = abs(output_i) / torch.norm(w_i.view(w_i.size(0), -1), 2, -1)

                # determine which w_i to use
                if pertur_i < pertur:
                    pertur = pertur_i
                    w = w_i

            # compute r_i and r_tot
            r_i = (pertur+1e-4) * w / torch.norm(w.view(w.size(0), -1), 2, -1)
            r_tot = r_tot + r_i

            pertur_imgs = images + (1+self.eps) * r_tot
            images = pertur_imgs.detach()

            images.requires_grad = True
            outputs = self.model(images)
            labels_flag = torch.argmax(outputs, -1)
            loop_i += 1

        return (1+self.eps)*r_tot, loop_i, pred_labels, labels_flag, pertur_imgs.detach()