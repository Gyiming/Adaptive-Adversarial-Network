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

parser = argparse.ArgumentParser(description='cifar10 Training')
#parser.add_argument('--gpu', default='7')
parser.add_argument('--gpu', default='1')
parser.add_argument('--cpus', default=4)
# dataset:
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'svhn', 'stl10'],
                    help='which dataset to use')
# optimization parameters:
parser.add_argument('--batch_size', '-b', default=64, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=1, type=int, help='number of total epochs to run')
parser.add_argument('--decay_epochs', '--de', default=[50, 150], nargs='+', type=int,
                    help='milestones for multisteps lr decay')
parser.add_argument('--opt', default='sgd', choices=['sgd', 'adam'], help='which optimizer to use')
parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
# adv parameters:
parser.add_argument('--targeted', action='store_true', help='If true, targeted attack')
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--steps', type=int, default=7)
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
#parser.add_argument('--resume', action='store_true', default=True,help='If true, resume from early stopped ckpt')
args = parser.parse_args()
args.efficient = True
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

# mkdirs:
#save_folder = '/u/ygan10/Once-for-All-Adversarial-Training/PGDAT/cifar10/ResNet34/8_simba/'
save_folder = '/u/ygan10/Once-for-All-Adversarial-Training/PGDAT2attack/cifar10/ResNet34/4_8'
print(save_folder)

# model:
if args.dataset == 'cifar10':
    model_fn = ResNet34
elif args.dataset == 'svhn':
    model_fn = WRN16_8
elif args.dataset == 'stl10':
    model_fn = WRN40_2

model = model_fn(num_classes = 100).cuda()
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
        
'''

# attacker:
attacker = PGD(eps=args.eps/255, steps = args.steps)
#attacker2 = PGD(eps=(args.eps+8)/255, steps = args.steps)
val_accs, val_accs_adv1, val_accs_adv2 = AverageMeter(), AverageMeter(), AverageMeter()

for i, (imgs, labels) in enumerate(val_loader):
    imgs, labels = imgs.cuda(), labels.cuda()
    
    for imagesave_idx in range(len(labels)):
        pkl_save_dir = '/u/ygan10/Once-for-All-Adversarial-Training/ori_bb_50/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num[labels[imagesave_idx]]) + '.pkl'
        with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
            pkl.dump(imgs[imagesave_idx].cpu().numpy(),f)
        pkl_save_dir = '/u/ygan10/Once-for-All-Adversarial-Training/ori_bb_50_ptolemy/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num[labels[imagesave_idx]]) + '.pkl'
        with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
            pkl.dump(imgs[imagesave_idx].detach().unsqueeze(0).permute(0,2,3,1).cpu().numpy(),f)
        img_num[labels[imagesave_idx]] += 1 

    with ctx_noparamgrad_and_eval(model):
        imgs_adv1 = attacker.attack(model, imgs, labels)
        #imgs_adv2 = attacker2.attack(model, imgs, labels)

    for imagesave_idx in range(len(labels)):
        pkl_save_dir = '/u/ygan10/Once-for-All-Adversarial-Training/pgd8_bb_50/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv1[labels[imagesave_idx]]) + '.pkl'
        with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
            pkl.dump(imgs_adv1[imagesave_idx].detach().cpu().numpy(),f)
        pkl_save_dir = '/u/ygan10/Once-for-All-Adversarial-Training/pgd8_bb_50_ptolemy/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv1[labels[imagesave_idx]]) + '.pkl'
        with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
            pkl.dump(imgs_adv1[imagesave_idx].detach().unsqueeze(0).permute(0,2,3,1).cpu().numpy(),f)            
        img_num_adv1[labels[imagesave_idx]] += 1

    
    for imagesave_idx in range(len(labels)):
        pkl_save_dir = '/u/ygan10/Once-for-All-Adversarial-Training/pgd16_bb/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv2[labels[imagesave_idx]]) + '.pkl'
        with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
            pkl.dump(imgs_adv2[imagesave_idx].detach().cpu().numpy(),f)
        pkl_save_dir = '/u/ygan10/Once-for-All-Adversarial-Training/pgd16_bb_50_ptolemy/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv2[labels[imagesave_idx]]) + '.pkl'
        with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
            pkl.dump(imgs_adv2[imagesave_idx].detach().unsqueeze(0).permute(0,2,3,1).cpu().numpy(),f)            
        img_num_adv2[labels[imagesave_idx]] += 1 
    
        logits_adv1 = model(imgs_adv1.detach())
        logits_adv2 = model(imgs_adv2.detach())
        # logits for clean imgs:
        logits = model(imgs)



    print(img_num)
    print(img_num_adv1)
    print(img_num_adv2) 
'''
