import os, argparse, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler

from models.cifar10.albation import ResNet34OAT
from models.svhn.wide_resnet_OAT import WRN16_8OAT
from models.stl10.wide_resnet_OAT import WRN40_2OAT

from dataloaders.cifar100 import cifar100_dataloaders
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
parser.add_argument('--gpu', default='1')
parser.add_argument('--cpus', default=4)
# dataset:
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'svhn', 'stl10'], help='which dataset to use')
# optimization parameters:
parser.add_argument('--batch_size', '-b', default=128, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=250, type=int, help='number of total epochs to run')
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
# OAT parameters:
parser.add_argument('--distribution', default='disc', choices=['disc'], help='Lambda distribution')
parser.add_argument('--lambda_choices', default=[0.0,0.1,0.2,0.3,0.4,1.0], nargs='*', type=float, help='possible lambda values to sample during training')
parser.add_argument('--probs', default=-1, type=float, help='the probs for sample 0, if not uniform sample')
parser.add_argument('--encoding', default='rand', choices=['none', 'onehot', 'dct', 'rand'], help='encoding scheme for Lambda')
parser.add_argument('--dim', default=128, type=int, help='encoding dimention for Lambda')
parser.add_argument('--use2BN', action='store_true', help='If true, use dual BN')
parser.add_argument('--sampling', default='ew', choices=['ew', 'bw'], help='sampling scheme for Lambda')
# others:
parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
args = parser.parse_args()
args.efficient = True
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True
args.use2BN = True
# data loader:
if args.dataset == 'cifar10':
    train_loader, val_loader, _ = cifar100_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'svhn':
    train_loader, val_loader, _ = svhn_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'stl10':
    train_loader, val_loader = stl10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)

# model:
if args.encoding in ['onehot', 'dct', 'rand']:
    FiLM_in_channels = args.dim
else: # non encoding
    FiLM_in_channels = 1
if args.dataset == 'cifar10':
    model_fn = ResNet34OAT
elif args.dataset == 'svhn':
    model_fn = WRN16_8OAT
elif args.dataset == 'stl10':
    model_fn = WRN40_2OAT

model = model_fn(use2BN=True, FiLM_in_channels=FiLM_in_channels, num_classes = 100).cuda()
model = torch.nn.DataParallel(model)

# mkdirs:
save_folder = '/u/ygan10/Once-for-All-Adversarial-Training/2BNDYN/cifar100/ResNet34OAT-2BN/albation_res1/'
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
#pdb.set_trace()

img_num = [0 for i in range(10)]
img_num_adv1 = [0 for i in range(10)]
img_num_adv2 = [0 for i in range(10)]

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


file_valid = []
file_valid = []
base_path = './testimg/test_single_c100_ori/'
for j in range(5000):
    file_valid.append(base_path + str(j) + '.pkl')
label_valid = np.load('./testimg/vali_single_c100.npy')
label_adv = np.load('./testimg/adv_single_c100.npy')
label_adv_gt = np.load('./testimg/adv_single_c100.npy')

validdata = traindataset(file_normal = file_valid, target = label_valid ,adv_label = label_adv,imgloader=imgloader,adv_label_gt=label_adv_gt)
valiloader = DataLoader(validdata, batch_size = args.batch_size, shuffle = True)
model.eval()
requires_grad_(model, False)

vector_normal = torch.from_numpy(encoding_mat[0]).unsqueeze(0).float().cuda()
vector_att1 = torch.from_numpy(encoding_mat[1]).unsqueeze(0).float().cuda()
vector_att2 = torch.from_numpy(encoding_mat[2]).unsqueeze(0).float().cuda()

val_accs, val_accs_adv1, val_accs_adv2 = AverageMeter(), AverageMeter(), AverageMeter()
correct_normal = 0
correct_adv = 0
total_normal = 0
total_adv = 0
correct_total = 0

for i, (imgs, labels, adv_labels, adv_labels_gt) in enumerate(valiloader):
    imgs, labels = imgs.cuda(), labels.cuda()
            
    index_normal = []
    index_adv1 = []
    index_adv2 = []
    index_normal_gt = []
    index_adv1_gt = []
    index_adv2_gt = []
    for t in range(len(adv_labels_gt)):
        if adv_labels_gt[t] == 0:
            index_normal_gt.append(t)
        elif adv_labels_gt[t] == 1:
            index_adv1_gt.append(t)
        elif adv_labels_gt[t] == 2:
            index_adv2_gt.append(t)
    
    for t in range(len(adv_labels)):
        if adv_labels[t] == 0:
            index_normal.append(t)
        elif adv_labels[t] == 1:
            index_adv1.append(t)
        elif adv_labels[t] == 2:
            index_adv2.append(t)
    imgs_normal = imgs[index_normal]
    labels_normal = labels[index_normal]
    imgs_adv1 = imgs[index_adv1]
    labels_adv1 = labels[index_adv1]
    imgs_adv2 = imgs[index_adv2]
    labels_adv2 = labels[index_adv2]

    #pdb.set_trace()
    if len(imgs_adv1) != 0:
        logits_adv1 = model(imgs_adv1.detach(), _lambda = vector_att1.repeat(imgs_adv1.size()[0],1,1), idx2BN = 0)
        val_accs_adv1.append((logits_adv1.argmax(1) == labels_adv1).float().mean().item())
        correct_total += val_accs_adv1.values[-1] * len(labels_adv1)
    if len(imgs_adv2) != 0:
        logits_adv2 = model(imgs_adv2.detach(), _lambda = vector_att2.repeat(imgs_adv2.size()[0],1,1), idx2BN = 1)
        val_accs_adv2.append((logits_adv2.argmax(1) == labels_adv2).float().mean().item())
        correct_total += val_accs_adv2.values[-1] * len(labels_adv2)
    if len(imgs_normal) != 0:
        logits_c = model(imgs_normal,_lambda = vector_normal.repeat(imgs_normal.size()[0],1,1), idx2BN = imgs_normal.size()[0])
        val_accs.append((logits_c.argmax(1) == labels_normal).float().mean().item())
        correct_total += val_accs.values[-1] * len(labels_normal)

    #pdb.set_trace()

pdb.set_trace()
val_str = 'SA: %.4f, RAtpgd6: %.4f, RAtpgd12: %.4f' % (val_accs.avg, val_accs_adv1.avg, val_accs_adv2.avg)
print(val_str)