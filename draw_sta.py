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
import matplotlib.pyplot as plt

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
save_folder = '/localdisk2/PGDAT/cifar10/ResNet34/noadv'
print(save_folder)
check_point = torch.load(os.path.join(save_folder,'best_SA.pth'))
#pdb.set_trace()
#pretrained_dict = check_point['state_dict']

means = {}
vars_ = {}
for name, param in check_point.items():
    print(name)
    if 'running_mean' in name:
        #pdb.set_trace()
        means[name] = []
        means[name].append(param.cpu().numpy())
    if 'running_var' in name:
        vars_[name] = []
        vars_[name].append(param.cpu().numpy())
#pdb.set_trace()
#layers = [i for i in range(len(means))]

save_folder = '/localdisk2/PGDAT/cifar10/ResNet34/onlyadv'
print(save_folder)
check_point = torch.load(os.path.join(save_folder,'best_RA1.pth'))
#pdb.set_trace()
#pretrained_dict = check_point['state_dict']

means2 = {}
vars2_ = {}
for name, param in check_point.items():
    print(name)
    if 'running_mean' in name:
        #pdb.set_trace()
        means2[name] = []
        means2[name].append(param.cpu().numpy())
    if 'running_var' in name:
        vars2_[name] = []
        vars2_[name].append(param.cpu().numpy())
#pdb.set_trace()

save_folder = '/localdisk2/PGDAT/cifar10/ResNet34/8'
print(save_folder)
check_point = torch.load(os.path.join(save_folder,'best_SA.pth'))
means3 = {}
vars3_ = {}
for name, param in check_point.items():
    print(name)
    if 'running_mean' in name:
        #pdb.set_trace()
        means3[name] = []
        means3[name].append(param.cpu().numpy())
    if 'running_var' in name:
        vars3_[name] = []
        vars3_[name].append(param.cpu().numpy())
#pdb.set_trace()

l1 = plt.scatter(means['module.layer3.2.bn2.running_mean'], vars_['module.layer3.2.bn2.running_var'], s=None, c='b', marker=None)
l2 = plt.scatter(means2['module.layer3.2.bn2.running_mean'], vars2_['module.layer3.2.bn2.running_var'], s=None, c='g', marker='*')
l3 = plt.scatter(means3['module.layer3.2.bn2.running_mean'], vars3_['module.layer3.2.bn2.running_var'], s=None, c='r', marker='^')
plt.legend(handles=[l1,l2,l3],labels=['std','onlyadv','adv'],loc='best')
plt.xlabel('running mean')
plt.ylabel('running var')
plt.savefig('adv_noadv_bn3.2.jpg')

'''
plt.plot(layers, means, color='blue')
plt.legend()
plt.xticks(layers)
plt.xlabel('layers')
'''