import torch
import torch.nn as nn
from attack import Attack
from pro_addm_batch import IPGDM
from class_label import classes

class UniversalAttack(Attack):

    def __init__(self, model, eps=0.150, delta=0.2, max_uni_iter=10,):
        super(UniversalAttack, self).__init__('UniversalAttack', model)
        self.eps = eps
        self.delta = delta
        self.max_uni_iter = max_uni_iter
        self.proximal = IPGDM(self.model)

    def forward(self, train_images_set, test_images_set, labels):
        
        train_images_set, test_images_set = train_images_set.to(self.device), test_images_set.to(self.device)

        pred_ori_outputs = self.model(test_images_set)
        pred_ori_labels = torch.argmax(pred_ori_outputs, 1)

        perturbation = torch.zeros_like(test_images_set[0]).unsqueeze(0)  # shape (num_segments*C, H, W)
        
        fooling_rate, iteration = 0.0, 0
        fooling_rates, total_iterations = [0], [0]

        pred_test_att_labels = pred_ori_labels
        iteration = 0

        adv_train_set = train_images_set + perturbation
        pred_train_labels = torch.argmax(self.model(train_images_set), dim=1)

        #while fooling_rate < 1-self.delta and iteration < self.max_uni_iter:
        while pred_test_att_labels.item() == pred_ori_labels.item() and iteration < self.max_uni_iter:
            print('Iteration:------', iteration)

            ############################################################################################################
            ################################################# generation ###############################################
            pred_adv_train_labels = torch.argmax(self.model(adv_train_set), dim=1)
            train_per = (pred_train_labels != pred_adv_train_labels).sum() / len(pred_train_labels)
            print('Train perturbed rate: ', train_per.item())
           
            adv_train_set, pertur, _, acc = self.proximal.forward(adv_train_set, labels, target=False)
            perturbation += pertur

            print('Train sucess attack rate: ', acc)
            iteration += 1

            ###########################################################################################################################
            ##################################################### verification ########################################################
            with torch.no_grad():

                # test_set shape (10, 3, 224, 224) # random sampling 10 samples to calculate fooling rate  

                pertur_inputs = test_images_set + perturbation
                per_outputs = self.model(pertur_inputs)
                pred_test_att_labels = torch.argmax(per_outputs, 1)
        
                fooling_rate = float(torch.sum(pred_test_att_labels != pred_ori_labels)) / float(len(test_images_set))
                success_rate = float(torch.sum(pred_test_att_labels == labels.to(self.device))) / float(len(test_images_set))

            print('=============================')
            print('Fooling rate: ', fooling_rate)
            print('Test success rate: ', success_rate)
            fooling_rates.append(fooling_rate)
            total_iterations.append(iteration)
        return perturbation, pred_ori_labels, pred_test_att_labels