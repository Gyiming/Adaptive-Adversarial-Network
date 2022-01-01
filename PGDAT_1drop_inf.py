import os, argparse, time
import numpy as np
import matplotlib.pyplot as plt
import pdb

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler

from models.cifar10.resnet_drop import ResNet34
from models.svhn.wide_resnet import WRN16_8
from models.stl10.wide_resnet import WRN40_2
from models.cifar10.RNet import ResNet50

from dataloaders.cifar10 import cifar10_dataloaders
from dataloaders.svhn import svhn_dataloaders
from dataloaders.stl10 import stl10_dataloaders

from utils.utils import *
from utils.context import ctx_noparamgrad_and_eval
from attacks.pgd import PGD

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb
import os
import PIL
from PIL import Image
import gzip
import pickle

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

parser = argparse.ArgumentParser(description='CIFAR10 Training against DDN Attack')
parser.add_argument('--gpu', default='1')
parser.add_argument('--cpus', default=4)
# dataset:
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'svhn', 'stl10'], help='which dataset to use')
# optimization parameters:
parser.add_argument('--batch_size', '-b', default=128, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=210, type=int, help='number of total epochs to run')
parser.add_argument('--decay_epochs', '--de', default=[50,150], nargs='+', type=int, help='milestones for multisteps lr decay')
parser.add_argument('--opt', default='sgd', choices=['sgd', 'adam'], help='which optimizer to use')
parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
# adv parameters:
parser.add_argument('--targeted', action='store_true', help='If true, targeted attack')
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--steps', type=int, default=7)
# loss parameters:
parser.add_argument('--Lambda', default=0.5, type=float, help='adv loss tradeoff parameter')
# others:
parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

# data loader:
if args.dataset == 'cifar10':
    train_loader, val_loader, _ = cifar10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'svhn':
    train_loader, val_loader, _ = svhn_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'stl10':
    train_loader, val_loader = stl10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)

# model:
if args.dataset == 'cifar10':
    #model_fn = ResNet34
    #model_fn = ResNet50
    model_fn = ResNet34
elif args.dataset == 'svhn':
    model_fn = WRN16_8
elif args.dataset == 'stl10':
    model_fn = WRN40_2
model = model_fn(num_classes = 100).cuda()
model = torch.nn.DataParallel(model)

# mkdirs:
model_str = model_fn.__name__
if args.opt == 'sgd':
    opt_str = 'e%d-b%d_sgd-lr%s-m%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.momentum, args.wd)
elif args.opt == 'adam':
    opt_str = 'e%d-b%d_adam-lr%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.wd)
if args.decay == 'cos':
    decay_str = 'cos'
elif args.decay == 'multisteps':
    decay_str = 'multisteps-%s' % args.decay_epochs
attack_str = 'targeted' if args.targeted else 'untargeted' + '-pgd-%d-%d' % (args.eps, args.steps)
loss_str = 'lambda%s' % (args.Lambda)

save_folder = '/u/ygan10/Once-for-All-Adversarial-Training/PGDAT/cifar10/ResNet34/dropout8'
print(save_folder)


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

# attacker:
attacker = PGD(eps=args.eps/255, steps=args.steps)
attacker2 = PGD(eps=(args.eps+6)/255, steps=args.steps)

file_valid = []
base_path = './testimg/test_single/'
for j in range(5000):
    file_valid.append(base_path + str(j) + '.pkl')
label_valid = np.load('./testimg/vali_single.npy')
label_adv = np.load('./testimg/adv_single.npy')
label_adv_gt = np.load('./testimg/adv_labels_3.npy')
validdata = traindataset(file_normal = file_valid, target = label_valid ,adv_label = label_adv,imgloader=imgloader,adv_label_gt=label_adv_gt)
valiloader = DataLoader(validdata, batch_size = args.batch_size, shuffle = True)
model.eval()
requires_grad_(model, False)

val_accs, val_accs_adv1, val_accs_adv2 = AverageMeter(), AverageMeter(), AverageMeter()
for i, (imgs, labels, adv_labels, adv_labels_gt) in enumerate(valiloader):
    imgs, labels = imgs.cuda(), labels.cuda()

    logits_all = model(imgs)
    #pdb.set_trace()
    normal_gt = []
    adv_gt_1 = []
    adv_gt_2 = []
    for t in range(len(adv_labels_gt)):
        if adv_labels_gt[t] == 0:
            normal_gt.append(t)
        elif adv_labels_gt[t] == 1:
            adv_gt_1.append(t)
        elif adv_labels_gt[t] == 2:
            adv_gt_2.append(t)
    #val_accs_adv1.append((logits_all[adv_gt_1].argmax(1) == labels[adv_gt_1]).float().mean().item())
    #val_accs_adv2.append((logits_all[adv_gt_2].argmax(1) == labels[adv_gt_2]).float().mean().item())
    val_accs.append((logits_all[normal_gt].argmax(1) == labels[normal_gt]).float().mean().item())

val_str = 'Acc: %.4f' % (val_accs.avg)   
print(val_str)

