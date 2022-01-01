import os, argparse, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler
import pdb

from models.imagenet.RNet import ResNet50
from dataloaders.imagenet import imagenet_dataloaders
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

def main():

    parser = argparse.ArgumentParser(description='cifar10 Training')
    #parser.add_argument('--gpu', default='7')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--cpus', default=16)
    # dataset:
    parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'svhn', 'stl10'],
                        help='which dataset to use')
    # optimization parameters:
    parser.add_argument('--batch_size', '-b', default=48, type=int, help='mini-batch size')
    parser.add_argument('--epochs', '-e', default=80, type=int, help='number of total epochs to run')
    parser.add_argument('--decay_epochs', '--de', default=[30, 70], nargs='+', type=int,
                        help='milestones for multisteps lr decay')
    parser.add_argument('--opt', default='adam', choices=['sgd', 'adam'], help='which optimizer to use')
    parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    # adv parameters:
    parser.add_argument('--targeted', action='store_true', help='If true, targeted attack')
    parser.add_argument('--eps', type=int, default=2)
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

    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.backends.cudnn.benchmark = True

    # data loader:
    if args.dataset == 'cifar10':
        train_loader, val_loader = imagenet_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
    
    model_fn = ResNet50
    model = model_fn(num_classes = 1000).cuda()
    model = torch.nn.DataParallel(model,device_ids = [0,1])

    # optimizer:
    if args.opt == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.decay == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.decay == 'multisteps':
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

    # mkdirs:
    save_folder = '/u/ygan10/Once-for-All-Adversarial-Training/PGDAT/Imagenet/ResNet50/adv'
    print(save_folder)
    create_dir(save_folder)

    model_fn = ResNet50
    model = model_fn(num_classes = 1000).cuda()
    model = torch.nn.DataParallel(model)

    attacker = PGD(eps=args.eps/255, steps = args.steps)

    training_loss, val_TA, val_ATA = [], [], []
    
    for epoch in range(0, args.epochs):
        fp = open(os.path.join(save_folder, 'val_log.txt'), 'a+')
        start_time = time.time()
        ## training:
        model.train()
        requires_grad_(model, True)
        accs, accs_adv1, losses = AverageMeter(), AverageMeter(), AverageMeter()
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            #pdb.set_trace()
            with ctx_noparamgrad_and_eval(model):
                adv_imgs = attacker.attack(model, imgs, labels)

            logits_adv1 = model(adv_imgs.detach())
            logits = model(imgs)

            # loss and update:
            loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits_adv1, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # proximal gradient for channel pruning:
            current_lr = scheduler.get_lr()[0]

            # metrics:
            accs.append((logits.argmax(1) == labels).float().mean().item())           
            accs_adv1.append((logits_adv1.argmax(1) == labels).float().mean().item())
            losses.append(loss.item())

            if i % 50 == 0:
                train_str = 'Epoch %d-%d | Train | Loss: %.4f, SA: %.4f' % (epoch, i, losses.avg, accs.avg)
                train_str += ', RA1: %.4f' % (accs_adv1.avg)
                #train_str += ', RA2: %.4f' % (accs_adv2.avg)
                print(train_str)
        # lr schedualr update at the end of each epoch:
        scheduler.step()

        model.eval()
        requires_grad_(model, False)
        print(model.training)

        if args.dataset == 'cifar10':
            eval_this_epoch = (epoch % 10 == 0) or (epoch>=int(0.7*args.epochs)) # boolean

        if eval_this_epoch:
            val_accs, val_accs_adv = AverageMeter(), AverageMeter()
            for i, (imgs, labels) in enumerate(val_loader):
                imgs, labels = imgs.cuda(), labels.cuda()

                # generate adversarial images:
                with ctx_noparamgrad_and_eval(model):
                    imgs_adv = attacker.attack(model, imgs, labels)
                    #imgs_adv2 = attacker2.attack(model, imgs, labels)
                linf_norms = (imgs_adv - imgs).view(imgs.size()[0], -1).norm(p=np.Inf, dim=1)
                logits_adv1 = model(imgs_adv.detach())
                #logits_adv2 = model(imgs_adv2.detach())
                # logits for clean imgs:
                logits = model(imgs)
                #pdb.set_trace()
                val_accs.append((logits.argmax(1) == labels).float().mean().item())
                val_accs_adv.append((logits_adv1.argmax(1) == labels).float().mean().item())
                #val_accs_adv2.append((logits_adv2.argmax(1) == labels).float().mean().item())

            val_str = 'Epoch %d | Validation | Time: %.4f | lr: %s | SA: %.4f, RA1: %.4f, linf: %.4f - %.4f' % (
                epoch, (time.time()-start_time), current_lr, val_accs.avg, val_accs_adv.avg,
                torch.min(linf_norms).data, torch.max(linf_norms).data)
            print(val_str)
            fp.write(val_str + '\n')

    # save loss curve:
    training_loss.append(losses.avg)
    plt.plot(training_loss)
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'training_loss.png'))
    plt.close()

    if eval_this_epoch:
        val_TA.append(val_accs.avg) 
        plt.plot(val_TA, 'r')
        val_ATA.append(val_accs_adv.avg)
        plt.plot(val_ATA, 'g')
        #val_ATA2.append(val_accs_adv2.avg)
        #plt.plot(val_ATA2, 'b')
        plt.grid(True)
        plt.savefig(os.path.join(save_folder, 'val_acc.png'))
        plt.close()
    else:
        val_TA.append(val_TA[-1]) 
        plt.plot(val_TA, 'r')
        val_ATA.append(val_ATA[-1])
        plt.plot(val_ATA, 'g')
        #val_ATA2.append(val_ATA2[-1])
        #plt.plot(val_ATA2, 'b')
        plt.grid(True)
        plt.savefig(os.path.join(save_folder, 'val_acc.png'))
        plt.close()

    # save pth:
    if eval_this_epoch:
        if val_accs.avg >= best_TA:
            best_TA = val_accs.avg
            torch.save(model.state_dict(), os.path.join(save_folder, 'best_SA.pth'))
        if val_accs_adv.avg >= best_ATA:
            best_ATA = val_accs_adv.avg
            torch.save(model.state_dict(), os.path.join(save_folder, 'best_RA.pth'))
    save_ckpt(epoch, model, optimizer, scheduler, best_TA, best_ATA, training_loss, val_TA, val_ATA, 
        os.path.join(save_folder, 'latest.pth'))

if __name__ == '__main__':
    main()
    



