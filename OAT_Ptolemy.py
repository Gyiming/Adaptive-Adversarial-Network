import os, argparse, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler

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

from skimage import io
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
    #pdb.set_trace()
    '''
    img_pil = Image.open(path)
    trans = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))])
    '''
    #img = torch.from_numpy(img_pil[0])
    #return img.permute(2,0,1)
    return torch.from_numpy(img_pil)

def vectorloader(path):
    vector = np.load(path)
    return torch.from_numpy(vector)

class mydataset(Dataset):
    def __init__(self, file_train,adv_train,target_train,vector_train,imgloader = imgloader, vectorloader = vectorloader):
        #定义好 image 的路径
        self.images = file_train
        self.adv = adv_train
        self.target = target_train
        self.vector = vector_train
        self.imgloader = imgloader
        self.vectorloader = vectorloader

    def __getitem__(self, index):
        img = self.images[index]
        adv = self.adv[index]
        target = self.target[index]
        vector = self.vector[index]
        return self.imgloader(img),adv,target,self.vectorloader(vector)

    def __len__(self):
        return len(self.images)

class traindataset(Dataset):
    def __init__(self, file_normal, file_adv, vec_normal, vec_adv, target_train, imgloader = imgloader, vectorloader = vectorloader):
        self.image_normal = file_normal
        self.image_adv = file_adv
        self.target = target_train
        self.vec_normal = vec_normal
        self.vec_adv = vec_adv
        self.imgloader = imgloader
        self.vectorloader = vectorloader

    def __getitem__(self,index):
        img_normal = self.image_normal[index]
        img_adv = self.image_adv[index]
        target = self.target[index]
        vector_normal = self.vec_normal[index]
        vector_adv = self.vec_adv[index]
        return self.imgloader(img_normal), self.imgloader(img_adv), target, self.vectorloader(vector_normal), self.vectorloader(vector_adv)

    def __len__(self):
        return len(self.image_normal)

def main():

    parser = argparse.ArgumentParser(description='cifar10 Training')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--cpus', default=4)
    # dataset:
    parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'svhn', 'stl10'], help='which dataset to use')
    # optimization parameters:
    parser.add_argument('--batch_size', '-b', default=128, type=int, help='mini-batch size')
    parser.add_argument('--epochs', '-e', default=200, type=int, help='number of total epochs to run')
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
    parser.add_argument('--use2BN', action='store_true', help='If true, use dual BN')
    parser.add_argument('--dim', default=18, type=int, help='length of feature vector')
    # others:
    parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')

    args = parser.parse_args()
    args.efficient = True
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.dataset == 'cifar10':
        model_fn = ResNet34OAT

    # dataloader:
    #train loader
    file_train_normal = []
    vector_train_normal = []
    file_train_adv = []
    vector_train_adv = []
    label_train = np.load('train_class_labels.npy')
    
    base_dir = './ptolemy_train_img/normal/'
    for i in range(len(label_train)):
        file_train_normal.append(base_dir + str(i) + '.pkl')

    base_dir = './ptolemy_train_img/adv/'
    for i in range(len(label_train)):
        file_train_adv.append(base_dir + str(i) + '.pkl')
    
    base_dir = './ptolemy_train_iouvector/normal/'
    for i in range(len(label_train)):
        vector_train_normal.append(base_dir + str(i) + '.npy')    

    base_dir = './ptolemy_train_iouvector/adv/'
    for i in range(len(label_train)):
        vector_train_adv.append(base_dir + str(i) + '.npy')            
    #adv_train = np.load('train_labels.npy')
    
    #adv_train = [1 for i in range(128)]
    #label_train = [1 for i in range(128)]

    traindata = traindataset(file_normal=file_train_normal, file_adv=file_train_adv, vec_normal=vector_train_normal, vec_adv=vector_train_adv, target_train=label_train)
    trainloader = DataLoader(traindata, batch_size = args.batch_size, shuffle = True)

    #validation loader
    file_valid = []
    vector_valid = []
    adv_valid = np.load('valid_labels.npy')
    label_valid = np.load('valid_class_labels.npy')

    base_dir = './ptolemy_valid_img/'
    for i in range(len(label_valid)):
        file_valid.append(base_dir + str(i) + '.pkl')
    
    base_dir = './ptolemy_valid_iouvector/'
    for i in range(len(label_valid)):
        vector_valid.append(base_dir + str(i) + '.npy')
          

    
    validdata = mydataset(file_train = file_valid, adv_train = adv_valid, target_train = label_valid, vector_train = vector_valid)
    valiloader = DataLoader(validdata, batch_size = args.batch_size, shuffle = True)
    #pdb.set_trace()
    # model:
    FiLM_in_channels = args.dim
    model = model_fn(use2BN=args.use2BN, FiLM_in_channels=FiLM_in_channels).cuda()

    # mkdirs:
    model_str = os.path.join(model_fn.__name__)
    if args.use2BN:
        model_str += '-2BN'
    if args.opt == 'sgd':
        opt_str = 'e%d-b%d_sgd-lr%s-m%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.momentum, args.wd)
    elif args.opt == 'adam':
        opt_str = 'e%d-b%d_adam-lr%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.wd)
    if args.decay == 'cos':
        decay_str = 'cos'
    elif args.decay == 'multisteps':
        decay_str = 'multisteps-%s' % args.decay_epochs
    attack_str = 'targeted' if args.targeted else 'untargeted' + '-pgd-%s-%d' % (args.eps, args.steps)
    save_folder = os.path.join('ptolemy_ckpt', model_str)
    cmd = 'mkdir ' + save_folder
    if not os.path.exists(save_folder):
        print(cmd)
        #pdb.set_trace()
        os.system(cmd)

    # optimizer:
    if args.opt == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.decay == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.decay == 'multisteps':
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

    # load ckpt:
    if args.resume:
        last_epoch, best_TA, best_ATA, training_loss, val_TA, val_ATA \
            = load_ckpt(model, optimizer, scheduler, os.path.join(save_folder, 'latest.pth'))
        start_epoch = last_epoch + 1
    else:
        start_epoch = 0
    # training curve lists:
        training_loss, val_TA, val_ATA, best_TA, best_ATA = [], [], [], 0, 0

    # attacker:
    attacker = PGD(eps=args.eps/255, steps=args.steps, use_FiLM=True)

    #training:
    for epoch in range(start_epoch, args.epochs):
        model.train()
        requires_grad_(model, True)
        start_time = time.time()
        accs, accs_adv, losses = AverageMeter(), AverageMeter(), AverageMeter()
        print(model.training)
        for i, (img_normal, img_adv, labels, vector_normal, vector_adv ) in enumerate(trainloader):
            #offload to GPU
            img_normal, img_adv, labels, vector_normal, vector_adv = img_normal.cuda(), img_adv.cuda(), labels.cuda(), vector_normal.cuda(), vector_adv.cuda()
            #pdb.set_trace()

            #inference normal images and adv images separately

            # logits for clean imgs:
            logits_c = model(img_normal.float(), vector_normal.float(), idx2BN = img_normal.size()[0])
            # clean loss:
            lc = F.cross_entropy(logits_c, labels, reduction = 'none')

            #logits for adv imgs:
            logits_a = model(img_adv.float(), vector_adv.float(), idx2BN = 0)
            # adv loss:
            la = F.cross_entropy(logits_a, labels, reduction = 'none')

            #wc = torch.mean(prob_normal)
            #wa = torch.mean(prob_adv)

             # loss and update:
            loss = torch.mean(lc+la)
            #loss = torch.mean(la)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # get current lr:
            current_lr = scheduler.get_lr()[0]

            # metrics:
            accs.append((logits_c.argmax(1) == labels).float().mean().item())
            accs_adv.append((logits_a.argmax(1) == labels).float().mean().item())
            losses.append(loss.item())

            if i % 100 == 0:
                train_str = 'Epoch %d-%d | Train | Loss: %.4f, TA: %.4f, ATA: %.4f' % (
                    epoch, i, losses.avg, accs.avg, accs_adv.avg)
                print(train_str)
        # lr schedualr update at the end of each epoch:
        scheduler.step()

        ## validation:
        model.eval()
        requires_grad_(model, False)
        print(model.training)

        eval_this_epoch = (epoch % 10 == 0) or (epoch>=int(0.75*args.epochs)) # boolean
        #eval_this_epoch = True
        if eval_this_epoch:
            val_accs, val_accs_adv = AverageMeter(), AverageMeter()

            for i, (imgs, adv_labels, labels, iou_vector) in enumerate(valiloader):
                #offload to GPU
                imgs,labels,iou_vector = imgs.cuda(), labels.cuda(), iou_vector.cuda()
                
                index_normal = []
                index_adv = []
                #separate adv examples and normal examples in one batch
                for index in range(len(adv_labels)):
                    #normal example
                    if adv_labels[index] == -1:
                        index_normal.append(index)
                    #adv example
                    elif adv_labels[index] == 1:
                        index_adv.append(index)
                
                imgs_normal = imgs[index_normal]
                labels_normal = labels[index_normal]
                iouvector_normal = iou_vector[index_normal]
                #prob_normal = prob[index_normal]

                imgs_adv = imgs[index_adv]
                labels_adv = labels[index_adv]
                iouvector_adv = iou_vector[index_adv]
                #prob_adv = prob[index_adv]     
                #pdb.set_trace()
                # logits for clean imgs:
                #pdb.set_trace()
                logits_c = model(imgs_normal.float(), iouvector_normal.float(), idx2BN = imgs.size()[0])        
                # logits for adv imgs:
                logits_a = model(imgs_adv.float(), iouvector_adv.float(), idx2BN = 0)  

                val_accs.append((logits_c.argmax(1) == labels_normal).float().mean().item())    
                val_accs_adv.append((logits_a.argmax(1) == labels_adv).float().mean().item())

        val_str = 'Epoch %d | Validation | Time: %.4f | lr: %s' % (epoch, (time.time()-start_time), current_lr)
        if eval_this_epoch:
            val_str += 'TA: %.4f, ATA: %.4f\n' % (val_accs.avg, val_accs_adv.avg)
        print(val_str)

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
            plt.grid(True)
            plt.savefig(os.path.join(save_folder, 'val_acc.png'))
            plt.close()
        # save pth:
        if eval_this_epoch:
            if val_accs.avg >= best_TA:
                best_TA = val_accs.avg
            torch.save(model.state_dict(), os.path.join(save_folder, 'best_TA.pth'))
            if val_accs_adv.avg >= best_ATA:
                best_ATA = val_accs_adv.avg
            torch.save(model.state_dict(), os.path.join(save_folder, 'best_ATA.pth'))
            save_ckpt(epoch, model, optimizer, scheduler, best_TA, best_ATA, training_loss, val_TA, val_ATA, os.path.join(save_folder, 'latest.pth'))

    



if __name__ == '__main__':
    main()
