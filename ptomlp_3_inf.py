import os, argparse, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler

from models.cifar10.ptolemymlp import ptomlp

from dataloaders.cifar10 import cifar10_dataloaders
from dataloaders.svhn import svhn_dataloaders
from dataloaders.stl10 import stl10_dataloaders

from utils.utils import *
from utils.context import ctx_noparamgrad_and_eval
from utils.sample_lambda import element_wise_sample_lambda, batch_wise_sample_lambda
from attacks.pgd import PGD

from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from thop import profile
import pandas as pd

import pdb
import os
import PIL
from PIL import Image
import gzip
import pickle
import numpy as np

def get_metric(file,layer,num):
    rank = 1
    summary = pd.read_csv(file)
    summary = summary.merge(
                summary,
                on=["class_id", "image_id"],
                how="inner",
                suffixes=("", "_normal"),
            )
    metric_list = []
    #pdb.set_trace()
    used_layers = 0
    for i in layer:
        metric_list.append(
            summary[f"{i}.overlap_size_in_class"]
            / summary[f"{i}.overlap_size_total"]
        )
        used_layers += 1
        if used_layers == num:
            break
    return np.stack(metric_list, axis=1), summary

def vectorprocess(filename):
    random_state = np.random.mtrand._rand
    ori_label = 0
    pgd8_label = 1
    pgd16_label = 1

    used_layers = ['conv2d/Conv2D',
                   'conv2d_2/Conv2D',
                   'conv2d_3/Conv2D',
                   'conv2d_4/Conv2D',
                   'conv2d_5/Conv2D',
                   'conv2d_7/Conv2D',
                   'conv2d_8/Conv2D',
                   'conv2d_9/Conv2D',
                   'conv2d_10/Conv2D',
                   'conv2d_12/Conv2D',
                   'conv2d_13/Conv2D',
                   'conv2d_14/Conv2D',
                   'conv2d_15/Conv2D',
                   'conv2d_17/Conv2D',
                   'conv2d_18/Conv2D',
                   'conv2d_19/Conv2D',
                   'conv2d_20/Conv2D',
                   'dense/MatMul']

    ori_metric, ori_summary = get_metric(filename, used_layers, 18)
    
    ori_metric[np.isinf(ori_metric)] = 0
    ori_metric[np.isnan(ori_metric)] = 0

    return ori_metric

class traindataset(Dataset):
    def __init__(self, vector, label):
        self.vector = vector
        self.label = label
        

    def __getitem__(self,index):
        vector_ = self.vector[index]
        label_ = self.label[index]
        #print(index, label_)
        return vector_, label_

    def __len__(self):
        return len(self.vector)

def main():

    normal_vector_train = vectorprocess('/u/ygan10/Once-for-All-Adversarial-Training/metrics/c100_tpgd_ori_train.csv')
    adv_vector_train = vectorprocess('/u/ygan10/Once-for-All-Adversarial-Training/metrics/c100_tpgd8_train.csv')
    normal_vector_vali = vectorprocess('/u/ygan10/Once-for-All-Adversarial-Training/metrics/c100_tpgd3_test.csv')
    #adv_vector_vali = vectorprocess('/u/ygan10/Once-for-All-Adversarial-Training/metrics/c100_tpgd8_test.csv')
    

    
    vali_vector = normal_vector_vali

    normal_label_train = [0 for i in range(len(normal_vector_train))]
    normal_label_vali = [0 for i in range(len(normal_vector_vali))]
    #adv_label_vali = [1 for i in range(len(adv_vector_vali))]

    train_label = normal_label_train
    vali_label = normal_label_vali

    save_folder = os.path.join('./ptomlpckpt')

    model_fn = ptomlp
    model = model_fn().cuda()

    save_folder2 = './ptomlpckpt'
    model.load_state_dict(torch.load(os.path.join(save_folder2,'best_3.pth')),True)


    validata = traindataset(vector = vali_vector, label =  vali_label)
    valiloader = DataLoader(validata, batch_size = 2048, shuffle = True)

    val_accs = AverageMeter()
    model.eval()
    requires_grad_(model, False)
    x = []
    print(model.training)
    for j, (vector, label) in enumerate(valiloader):
        vector, label = vector.cuda(), label.cuda()
        logits = model(vector.float())
        logits_npy = logits.argmax(1).cpu().numpy()
        for i in range(len(logits_npy)):
            x.append(logits_npy[i])
        pdb.set_trace()
        val_accs.append((logits.argmax(1) == label).float().mean().item())
    #pdb.set_trace()
    print(val_accs.avg)
    #np.save('./testimg/joint_tpgd15_3.npy',x)



if __name__ == '__main__':
    main()