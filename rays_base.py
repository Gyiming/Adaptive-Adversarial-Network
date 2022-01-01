from general_torch_model import GeneralTorchModel
from RayS import RayS
import argparse
import time
import numpy as np
import os
import utils_square
from datetime import datetime

import os, argparse, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler
import pdb

from models.cifar10.resnet import ResNet34
from models.svhn.wide_resnet import WRN16_8
from models.stl10.wide_resnet import WRN40_2
from models.cifar10.RNet import ResNet50

from dataloaders.cifar10 import cifar10_dataloaders
from dataloaders.svhn import svhn_dataloaders
from dataloaders.stl10 import stl10_dataloaders

from utils.utils import *
from utils.context import ctx_noparamgrad_and_eval
from utils.sample_lambda import element_wise_sample_lambda, batch_wise_sample_lambda
from attacks.pgd import PGD
from torchvision.utils import save_image
import gzip
import pickle as pkl

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb
import os
import PIL
from PIL import Image
import gzip
import pickle

np.set_printoptions(precision=5, suppress=True)

def imgloader(path):
    #a = np.array([])
    with gzip.open(path,"rb") as file:
        img_pil = pickle.load(file)

    return torch.from_numpy(img_pil)

class traindataset(Dataset):
    def __init__(self, file_normal,target,adv_label,imgloader,adv_label_gt):
        self.image_normal = file_normal
        self.target = target
        self.adv_label = adv_label
        self.imgloader = imgloader
        self.adv_label_gt = adv_label_gt

    def __getitem__(self,index):
        img_normal = self.imgloader(self.image_normal[index])
        target = self.target[index]
        adv_label = self.adv_label[index]
        adv_label_gt = self.adv_label_gt[index]
        return img_normal, target, adv_label, adv_label_gt

    def __len__(self):
        return len(self.image_normal)

def main():
    parser = argparse.ArgumentParser(description='cifar10 Training')
    #parser.add_argument('--gpu', default='7')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--cpus', default=4)

    # dataset:
    parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'svhn', 'stl10'],
                        help='which dataset to use')
    # optimization parameters:
    parser.add_argument('--batch_size', '-b', default=256, type=int, help='mini-batch size')
    parser.add_argument('--epochs', '-e', default=1, type=int, help='number of total epochs to run')
    parser.add_argument('--decay_epochs', '--de', default=[50, 150], nargs='+', type=int,
                        help='milestones for multisteps lr decay')
    parser.add_argument('--opt', default='sgd', choices=['sgd', 'adam'], help='which optimizer to use')
    parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    # OAT parameters:
    parser.add_argument('--distribution', default='disc', choices=['disc'], help='Lambda distribution')
    parser.add_argument('--lambda_choices', default=[0.0, 0.1, 0.2, 0.3, 0.4, 1.0], nargs='*', type=float,
                        help='possible lambda values to sample during training')
    parser.add_argument('--probs', default=-1, type=float, help='the probs for sample 0, if not uniform sample')
    parser.add_argument('--encoding', default='rand', choices=['none', 'onehot', 'dct', 'rand'],
                        help='encoding scheme for Lambda')
    parser.add_argument('--dim', default=128, type=int, help='encoding dimention for Lambda')
    parser.add_argument('--use2BN', action='store_true', help='If true, use dual BN')
    parser.add_argument('--sampling', default='ew', choices=['ew', 'bw'], help='sampling scheme for Lambda')
    # others:
    parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
    parser.add_argument('--epsilon', default=0.05, type=float,help='attack strength')
    #parser.add_argument('--resume', action='store_true', default=True,help='If true, resume from early stopped ckpt')

    args = parser.parse_args()
    args.efficient = True
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.backends.cudnn.benchmark = True

    # data loader:
    if args.dataset == 'cifar10':
        train_loader, val_loader, _ = cifar10_dataloaders(train_batch_size=args.batch_size, test_batch_size=args.batch_size,num_workers=args.cpus)
    elif args.dataset == 'svhn':
        train_loader, val_loader, _ = svhn_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
    elif args.dataset == 'stl10':
        train_loader, val_loader = stl10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)

    # mkdirs:
    save_folder = '/u/ygan10/Once-for-All-Adversarial-Training/PGDAT/cifar10/ResNet34/8/'
    print(save_folder)

    # model:
    if args.dataset == 'cifar10':
        model_fn = ResNet34
    elif args.dataset == 'svhn':
        model_fn = WRN16_8
    elif args.dataset == 'stl10':
        model_fn = WRN40_2

    model = model_fn(num_classes = 10).cuda()
    model = torch.nn.DataParallel(model)

    # optimizer:
    if args.opt == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.decay == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.decay == 'multisteps':
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

    model.load_state_dict(torch.load(os.path.join(save_folder,'best_SA.pth')),True)
    #pdb.set_trace()

    img_num = [0 for i in range(10)]
    img_num_adv1 = [0 for i in range(10)]
    img_num_adv2 = [0 for i in range(10)]


    file_valid = []
    '''
    base_path = './testimg/testtpgd3_8_16/'
    for j in range(15000):
        file_valid.append(base_path + str(j) + '.pkl')
    label_valid = np.load('./testimg/vali_labels_3.npy')
    label_adv = np.load('./testimg/adv_labels_3.npy')
    label_adv_gt = np.load('./testimg/adv_labels_3.npy')
    validdata = traindataset(file_normal = file_valid, target = label_valid ,adv_label = label_adv,imgloader=imgloader,adv_label_gt=label_adv_gt)
    valiloader = DataLoader(validdata, batch_size = args.batch_size, shuffle = True)
    '''
    model.eval()
    requires_grad_(model, False)
    val_accs, val_accs_square = AverageMeter(), AverageMeter()
    #pdb.set_trace()
    torch_model = GeneralTorchModel(model, n_class=10, im_mean=[0.5, 0.5, 0.5], im_std=[0.5, 0.5, 0.5])
    attack = RayS(torch_model, epsilon=args.epsilon)

    for i, (imgs, labels) in enumerate(val_loader):
        imgs, labels = imgs.cuda(), labels.cuda()
        #pdb.set_trace()
        logits_clean = model(imgs)
        val_accs.append((logits_clean.argmax(1) == labels).float().mean().item())
        x_adv, queries, adbd, succ = attack(imgs, labels, query_limit = 2000)
        logits_square = model(x_adv)
        pdb.set_trace()
        val_accs_square.append((logits_square.argmax(1) == labels).float().mean().item())
        print('batch:',i)
    pdb.set_trace()
    val_str = 'SA: %.4f, RAsquare: %.4f' % (val_accs.avg, val_accs_square.avg)   
    print(val_str)

if __name__ == '__main__':
    main()