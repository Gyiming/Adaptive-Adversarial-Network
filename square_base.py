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

from dataloaders.cifar100 import cifar100_dataloaders
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

def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p

def pseudo_gaussian_pert_rectangles(x, y):
    delta = np.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1

    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
              max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

        counter2[0] -= 1
        counter2[1] -= 1

    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def meta_pseudo_gaussian_pert(s):
    delta = np.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        if np.random.rand(1) > 0.5: delta = np.transpose(delta)

    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta

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

def myloss(y, logits, targeted = False, loss_type = 'margin_loss'):
    #pdb.set_trace()
    if loss_type == 'margin_loss':
        preds_correct_class = (logits * y).sum(1, keepdims=True)
        diff = preds_correct_class - logits  # difference between the correct class and all other classes
        if y.shape[0] < 32:
            pdb.set_trace()
        diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
        margin = diff.min(1, keepdims=True)
        loss = margin * -1 if targeted else margin.values
    elif loss_type == 'cross_entropy':
            probs = utils.softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
    return loss.flatten()

def square_attack_linf(model, x, y, corr_classified, eps, n_iters, p_init, metrics_path, targeted, loss_type):
    np.random.seed(0)  # important to leave it here as well
    #pdb.set_trace()
    min_val, max_val = 0, 1 
    c, h, w = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]
    x, y = x[corr_classified], y[corr_classified]

    # [c, 1, w], i.e. vertical stripes work best for untargeted attacks
    init_delta = torch.from_numpy(np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])).to(x.device).float()
    x_best = torch.clamp(x + init_delta, min_val, max_val)

    logits = model(x_best)
    loss_min = myloss(y, logits.cpu(), targeted, loss_type)
    margin_min = myloss(y, logits.cpu(), targeted, loss_type)
    n_queries = torch.ones(x.shape[0]).to(x.device)  # ones because we have already used 1 query

    time_start = time.time()
    metrics = torch.zeros([n_iters, 7]).to(x.device)
    for i_iter in range(n_iters - 1):
        idx_to_fool = margin_min > 0
        x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
        loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
        deltas = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while torch.sum(torch.abs(torch.clamp(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = torch.from_numpy(np.random.choice([-eps, eps], size=[c, 1, 1])).to(deltas.device).float()

        x_new = torch.clamp(x_curr + deltas, min_val, max_val)

        logits = model(x_new)
        loss = myloss(y_curr, logits.cpu(), targeted, loss_type)
        margin = myloss(y_curr, logits.cpu(), targeted, loss_type)

        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = torch.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        #pdb.set_trace()
        acc = (margin_min > 0.0).sum().item() / n_ex_total
        acc_corr = (margin_min > 0.0).float().mean()
        mean_nq, mean_nq_ae = torch.mean(n_queries), torch.mean(n_queries[margin_min <= 0])
        avg_margin_min = torch.mean(margin_min)
        time_total = time.time() - time_start
        #print('{}: acc={:.2%} acc_corr={:.2%} avg #query={:.2f} avg #query_adv={:.2f} avg_margin={:.2f} (n_ex={}, eps={:.3f}, {:.2f}s)'.
        #    format(i_iter+1, acc, acc_corr, mean_nq, mean_nq_ae, avg_margin_min, x.shape[0], eps, time_total))

        if acc == 0:
            break

        # if mean_nq >= mean_nq_limit:
        #     print('out of mean query number limit')
        #     break

    return n_queries, x_best, acc

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
    #parser.add_argument('--resume', action='store_true', default=True,help='If true, resume from early stopped ckpt')

    #attack
    parser.add_argument('--n_ex', type=int, default=10000, help='Number of test ex to test on.')
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of changing a coordinate. Note: check the paper for the best values. '
                             'Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
    parser.add_argument('--eps', type=float, default=0.031, help='Radius of the Lp ball.')
    #eps=8 -> 0.031, eps=16 -> 0.063, eps=4 -> 0.016
    parser.add_argument('--n_iter', type=int, default=500)


    args = parser.parse_args()
    args.efficient = True
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.backends.cudnn.benchmark = True

    # data loader:
    if args.dataset == 'cifar10':
        train_loader, val_loader, _ = cifar100_dataloaders(train_batch_size=args.batch_size, test_batch_size=args.batch_size,num_workers=args.cpus)
    elif args.dataset == 'svhn':
        train_loader, val_loader, _ = svhn_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
    elif args.dataset == 'stl10':
        train_loader, val_loader = stl10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)

    # mkdirs:
    save_folder = '/u/ygan10/Once-for-All-Adversarial-Training/PGDAT/cifar100/ResNet34/8/'
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

    for i, (imgs, labels) in enumerate(val_loader):
        imgs, labels = imgs.cuda(), labels.cuda()
        #pdb.set_trace()
        logits_clean = model(imgs)
        val_accs.append((logits_clean.argmax(1) == labels).float().mean().item())
        corr_classified = logits_clean.argmax(1) == labels
        #print('correct predicted',corr_classified)
        square_attack = square_attack_linf

        metrics_path = 'metric.npy'
        y_target_onehot = utils_square.dense_to_onehot(labels.cpu(), n_cls=100)
        n_queries, x_adv, acc_after_attack = square_attack(model, imgs.cpu(), y_target_onehot, corr_classified.cpu(), args.eps, args.n_iter, args.p, metrics_path, False, 'margin_loss')
        logits_square = model(x_adv)
        #pdb.set_trace()
        val_accs_square.append(acc_after_attack)
        print('batch:',i)

    val_str = 'SA: %.4f, RAsquare: %.4f' % (val_accs.avg, val_accs_square.avg)   
    print(val_str)

if __name__ == '__main__':
    main()