import os, argparse, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler
import pdb

from models.cifar10.resnet_OAT import ResNet34OAT
from models.svhn.wide_resnet_OAT import WRN16_8OAT
from models.stl10.wide_resnet_OAT import WRN40_2OAT

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


parser = argparse.ArgumentParser(description='cifar10 Training')
#parser.add_argument('--gpu', default='7')
parser.add_argument('--gpu', default='0')
parser.add_argument('--cpus', default=4)
# dataset:
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'svhn', 'stl10'],
                    help='which dataset to use')
# optimization parameters:
parser.add_argument('--batch_size', '-b', default=128, type=int, help='mini-batch size')
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
save_folder = '/u/ygan10/Once-for-All-Adversarial-Training/2attack/cifar10/ResNet34OAT-2BN/8_16_3BN'
print(save_folder)

# encoding
if args.encoding == 'onehot':
    I_mat = np.eye(args.dim)
    encoding_mat = I_mat
elif args.encoding == 'dct':
    from scipy.fftpack import dct
    dct_mat = dct(np.eye(args.dim), axis=0)
    encoding_mat = dct_mat
elif args.encoding == 'rand':
    rand_mat = np.random.randn(args.dim, args.dim)
    np.save(os.path.join(save_folder, 'rand_mat.npy'), rand_mat)
    rand_otho_mat, _ = np.linalg.qr(rand_mat)
    np.save(os.path.join(save_folder, 'rand_otho_mat.npy'), rand_otho_mat)
    encoding_mat = rand_otho_mat
elif args.encoding == 'none':
    encoding_mat = None

# model:
if args.encoding in ['onehot', 'dct', 'rand']:
    FiLM_in_channels = args.dim
else:  # non encoding
    FiLM_in_channels = 1
if args.dataset == 'cifar10':
    model_fn = ResNet34OAT
elif args.dataset == 'svhn':
    model_fn = WRN16_8OAT
elif args.dataset == 'stl10':
    model_fn = WRN40_2OAT
model = model_fn(use2BN=args.use2BN, FiLM_in_channels=FiLM_in_channels).cuda()
model = torch.nn.DataParallel(model)
# for name, p in model.named_parameters():
#     print(name, p.size())



# optimizer:
if args.opt == 'sgd':
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
elif args.opt == 'adam':
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
if args.decay == 'cos':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
elif args.decay == 'multisteps':
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

#last_epoch, best_TA, best_ATA, training_loss, val_TA, val_ATA = load_ckpt(model, optimizer, scheduler, os.path.join(save_folder, 'best_RA2.pth'))
model.load_state_dict(torch.load(os.path.join(save_folder,'best_SA.pth')),False)
#a = torch.load(os.path.join(save_folder,'best_RA2.pth'))
pdb.set_trace()
attacker = PGD(eps=args.eps/255, steps = args.steps, use_FiLM = True)
attacker2 = PGD(eps=(args.eps+8)/255, steps = args.steps, use_FiLM = True)

model.eval()
requires_grad_(model, False)

vector_normal = torch.from_numpy(encoding_mat[0]).unzqueeze(0).float().cuda()
vector_att1 = torch.from_numpy(encoding_mat[1]).unzqueeze(0).float().cuda()
vector_att2 = torch.from_numpy(encoding_mat[2]).unzqueeze(0).float().cuda()

img_num = [0 for i in range(10)]
img_num_adv1 = [0 for i in range(10)]
img_num_adv2 = [0 for i in range(10)]

for epoch in range(1):
    for i, (imgs, laabels) in enumerate(train_loader):
        imgs, labels = imgs.cuda(), labels.cuda()

        for imagesave_idx in range(len(labels)):
            #pdb.set_trace()
            img_save_dir = '/u/ygan10/Once-for-All-Adversarial-Training-master/ori_train/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num[labels[imagesave_idx]]) + '.png'
            save_image(imgs[imagesave_idx].cpu(),img_save_dir)
            pkl_save_dir = '/u/ygan10/Once-for-All-Adversarial-Training-master/ori_train_pkl/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num[labels[imagesave_idx]]) + '.pkl'
            with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
                pkl.dump(imgs[imagesave_idx].cpu().numpy(),f)
            pkl_save_dir = '/u/ygan10/Once-for-All-Adversarial-Training-master/ori_train_ptolemy/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num[labels[imagesave_idx]]) + '.pkl'
            with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
                pkl.dump(imgs[imagesave_idx].detach().unsqueeze(0).permute(0,2,3,1).cpu().numpy(),f)
            img_num[labels[imagesave_idx]] += 1 

        with ctx_noparamgrad_and_eval(model):
            imgs_adv1 = attacker.attack(model, imgs, labels=labels, _lambda=vector_att1.repeat(imgs.size()[0],1,1), idx2BN=0)
            imgs_adv2 = attacker2.attack(model, imgs, labels=labels, _lambda=vector_att2.repeat(imgs.size()[0],1,1), idx2BN=1)

        for imagesave_idx in range(len(labels)):
            #pdb.set_trace()
            
            img_save_dir = '/u/ygan10/Once-for-All-Adversarial-Training-master/pgd8_train/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv1[labels[imagesave_idx]]) + '.png'
            save_image(imgs_adv1[imagesave_idx].cpu(),img_save_dir)
            pkl_save_dir = '/u/ygan10/Once-for-All-Adversarial-Training-master/pgd8_train/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv1[labels[imagesave_idx]]) + '.pkl'
            with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
                pkl.dump(imgs_adv1[imagesave_idx].detach().cpu().numpy(),f)
            pkl_save_dir = 'u/ygan10/Once-for-All-Adversarial-Training-master/pgd8_train_ptolemy/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv1[labels[imagesave_idx]]) + '.pkl'
            with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
                pkl.dump(imgs_adv1[imagesave_idx].detach().unsqueeze(0).permute(0,2,3,1).cpu().numpy(),f)            
            img_num_adv1[labels[imagesave_idx]] += 1


        for imagesave_idx in range(len(labels)):
            #pdb.set_trace()
            
            img_save_dir = '/u/ygan10/Once-for-All-Adversarial-Training-master/pgd16_train/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv2[labels[imagesave_idx]]) + '.png'
            save_image(imgs_adv2[imagesave_idx].cpu(),img_save_dir)
            pkl_save_dir = '/u/ygan10/Once-for-All-Adversarial-Training-master/pgd16_train/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv2[labels[imagesave_idx]]) + '.pkl'
            with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
                pkl.dump(imgs_adv2[imagesave_idx].detach().cpu().numpy(),f)
            pkl_save_dir = '/u/ygan10/Once-for-All-Adversarial-Training-master/pgd16_train_ptolemy/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv2[labels[imagesave_idx]]) + '.pkl'
            with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
                pkl.dump(imgs_adv2[imagesave_idx].detach().unsqueeze(0).permute(0,2,3,1).cpu().numpy(),f)            
            img_num_adv2[labels[imagesave_idx]] += 1     

        print(img_num)
        print(img_num_adv1)
        print(img_num_adv2) 

    img_num = [0 for i in range(10)]
    img_num_adv1 = [0 for i in range(10)]
    img_num_adv2 = [0 for i in range(10)]
    for i, (imgs, laabels) in enumerate(vali_loader):
        imgs, labels = imgs.cuda(), labels.cuda()

        for imagesave_idx in range(len(labels)):
            #pdb.set_trace()
            img_save_dir = '/home/Yangtze/xushengji/Once-for-All-Adversarial-Training-master/ori_vali/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num[labels[imagesave_idx]]) + '.png'
            save_image(imgs[imagesave_idx].cpu(),img_save_dir)
            pkl_save_dir = '/home/Yangtze/xushengji/Once-for-All-Adversarial-Training-master/ori_vali_pkl/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num[labels[imagesave_idx]]) + '.pkl'
            with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
                pkl.dump(imgs[imagesave_idx].cpu().numpy(),f)
            pkl_save_dir = '/home/Yangtze/xushengji/Once-for-All-Adversarial-Training-master/ori_vali_ptolemy/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num[labels[imagesave_idx]]) + '.pkl'
            with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
                pkl.dump(imgs[imagesave_idx].detach().unsqueeze(0).permute(0,2,3,1).cpu().numpy(),f)
            img_num[labels[imagesave_idx]] += 1 

        with ctx_noparamgrad_and_eval(model):
            imgs_adv1 = attacker.attack(model, imgs, labels=labels, _lambda=vector_att1.repeat(imgs.size()[0],1,1), idx2BN=0)
            imgs_adv2 = attacker2.attack(model, imgs, labels=labels, _lambda=vector_att2.repeat(imgs.size()[0],1,1), idx2BN=1)

        for imagesave_idx in range(len(labels)):
            #pdb.set_trace()
            
            img_save_dir = '/home/Yangtze/xushengji/Once-for-All-Adversarial-Training-master/pgd8_vali/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv1[labels[imagesave_idx]]) + '.png'
            save_image(imgs_adv1[imagesave_idx].cpu(),img_save_dir)
            pkl_save_dir = '/home/Yangtze/xushengji/Once-for-All-Adversarial-Training-master/pgd8_vali/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv1[labels[imagesave_idx]]) + '.pkl'
            with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
                pkl.dump(imgs_adv1[imagesave_idx].detach().cpu().numpy(),f)
            pkl_save_dir = '/home/Yangtze/xushengji/Once-for-All-Adversarial-Training-master/pgd8_vali_ptolemy/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv1[labels[imagesave_idx]]) + '.pkl'
            with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
                pkl.dump(imgs_adv1[imagesave_idx].detach().unsqueeze(0).permute(0,2,3,1).cpu().numpy(),f)            
            img_num_adv1[labels[imagesave_idx]] += 1


        for imagesave_idx in range(len(labels)):
            #pdb.set_trace()
            
            img_save_dir = '/home/Yangtze/xushengji/Once-for-All-Adversarial-Training-master/pgd16_vali/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv2[labels[imagesave_idx]]) + '.png'
            save_image(imgs_adv2[imagesave_idx].cpu(),img_save_dir)
            pkl_save_dir = '/home/Yangtze/xushengji/Once-for-All-Adversarial-Training-master/pgd16_vali/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv2[labels[imagesave_idx]]) + '.pkl'
            with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
                pkl.dump(imgs_adv2[imagesave_idx].detach().cpu().numpy(),f)
            pkl_save_dir = '/home/Yangtze/xushengji/Once-for-All-Adversarial-Training-master/pgd16_vali_ptolemy/' + str(labels[imagesave_idx].cpu().numpy()) + '/' + str(img_num_adv2[labels[imagesave_idx]]) + '.pkl'
            with gzip.open(pkl_save_dir,'wb',compresslevel=6) as f:
                pkl.dump(imgs_adv2[imagesave_idx].detach().unsqueeze(0).permute(0,2,3,1).cpu().numpy(),f)            
            img_num_adv2[labels[imagesave_idx]] += 1     

        print(img_num)
        print(img_num_adv1)
        print(img_num_adv2) 

      
