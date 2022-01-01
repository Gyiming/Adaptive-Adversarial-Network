''' PyTorch implementation of ResNet taken from 
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
and used by 
https://github.com/TAMU-VITA/ATMC/blob/master/cifar/resnet/resnet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.FiLM import FiLM_Layer
from models.DualBN import DualBN2d
import pdb

class BasicBlockOAT(nn.Module):

    def __init__(self, in_planes, mid_planes, out_planes, stride=1, use2BN=False, FiLM_in_channels=1):
        super(BasicBlockOAT, self).__init__()
        self.use2BN = use2BN
        if self.use2BN:
            Norm2d = DualBN2d
        else:
            Norm2d = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = Norm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = Norm2d(out_planes)

        if stride != 1 or in_planes != out_planes:
            self.mismatch = True
            self.conv_sc = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            self.bn_sc = Norm2d(out_planes)
        else:
            self.mismatch = False
        
        #self.film1 = FiLM_Layer(channels=mid_planes, in_channels=FiLM_in_channels) 
        #self.film2 = FiLM_Layer(channels=out_planes, in_channels=FiLM_in_channels)

    def forward(self, x, idx2BN=None):
        out = self.conv1(x)
        if self.use2BN:
            out = self.bn1(out, idx2BN)
        else:
            out = self.bn1(out)
        #out = self.film1(out, _lambda)
        out = F.relu(out)
        out = self.conv2(out)
        if self.use2BN:
            out = self.bn2(out, idx2BN)
        else:
            out = self.bn2(out)
        #out = self.film2(out, _lambda)
        
        if self.mismatch:
            if self.use2BN: 
                out += self.bn_sc(self.conv_sc(x), idx2BN)
            else:
                out += self.bn_sc(self.conv_sc(x))
        else:
            out += x
        out = F.relu(out)
        # print(out.size())
        return out

class BasicBlock(nn.Module):

    def __init__(self, in_planes, mid_planes, out_planes, stride=1, use2BN=False, FiLM_in_channels=1):
        super(BasicBlock, self).__init__()
        self.use2BN = use2BN
        Norm2d = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = Norm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = Norm2d(out_planes)

        if stride != 1 or in_planes != out_planes:
            self.mismatch = True
            self.conv_sc = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            self.bn_sc = Norm2d(out_planes)
        else:
            self.mismatch = False
        
        #self.film1 = FiLM_Layer(channels=mid_planes, in_channels=FiLM_in_channels) 
        #self.film2 = FiLM_Layer(channels=out_planes, in_channels=FiLM_in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.film1(out, _lambda)
        post_bn1 = out
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        post_bn2 = out
        #out = self.film2(out, _lambda)
        
        if self.mismatch:
            out += self.bn_sc(self.conv_sc(x))
        else:
            out += x
        out = F.relu(out)
        # print(out.size())
        return out, post_bn1, post_bn2

class ResNet34OAT(nn.Module):
    '''
    GFLOPS: 1.1837, model size: 31.4040MB
    '''
    def __init__(self, num_classes=10, FiLM_in_channels=1, use2BN=False):
        super(ResNet34OAT, self).__init__()
        self.use2BN = use2BN

        #self.convf1a1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        #self.convf1a2 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        #self.convf1c = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #if self.use2BN:
        #    self.bn1 = DualBN2d(64)
        #else:
        self.bn1 = nn.BatchNorm2d(64)
        #self.film1 = FiLM_Layer(channels=64, in_channels=FiLM_in_channels)
        self.bundle1 = nn.ModuleList([
            BasicBlock(64, 64, 64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlock(64, 64, 64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlock(64, 64, 64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.detect1 = nn.Conv2d(64,96,kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.max1 = nn.MaxPool2d(2, stride=1)
        self.detectbn1 = nn.BatchNorm2d(96)
        self.detect2 = nn.Conv2d(96,192,kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.max2 = nn.MaxPool2d(2, stride=1)
        self.detectbn2 = nn.BatchNorm2d(192)
        self.detect3 = nn.Conv2d(192,192,kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.detectbn3 = nn.BatchNorm2d(192)
        self.detect4 = nn.Conv2d(192,32,kernel_size = 1, stride = 1, padding = 1, bias = False)
        self.detectbn4 = nn.BatchNorm2d(32)
        self.detect_l1 = nn.Linear(2048,128)
        torch.nn.init.xavier_uniform(self.detect_l1.weight)
        self.detect_l1.bias.data.fill_(0.01)
        self.detect_l2 = nn.Linear(128,2)
        torch.nn.init.xavier_uniform(self.detect_l2.weight)
        self.detect_l2.bias.data.fill_(0.01)        
        self.bundle2 = nn.ModuleList([
            BasicBlockOAT(64, 128, 128, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(128, 128, 128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(128, 128, 128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(128, 128, 128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.bundle3 = nn.ModuleList([
            BasicBlockOAT(128, 256, 256, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.bundle4 = nn.ModuleList([
            BasicBlockOAT(256, 512, 512, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(512, 512, 512, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(512, 512, 512, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.linear = nn.Linear(512, num_classes)
        self.bundles = [self.bundle1, self.bundle2, self.bundle3, self.bundle4]


    def forward(self, x, idx2BN=None, fixed = False):
        out = self.conv1(x)
        out = self.bn1(out)

        out = F.relu(out)

        post_bn = []
        for block in self.bundles[0]:
            out, post_bn1, post_bn2 = block(out)
            post_bn.append(post_bn1)
            post_bn.append(post_bn2)
        '''
        detector_input = post_bn[0]
        for i in range(1,len(post_bn)):
            detector_input = torch.cat((detector_input,post_bn[i]),axis=1)
        '''
        #pdb.set_trace()
        d_ = self.detect1(post_bn[-1])
        d_ = self.detectbn1(d_)
        d_ = F.relu(d_)
        d_ = self.max1(d_)
        d_ = self.detect2(d_)
        d_ = self.detectbn2(d_)
        d_ = F.relu(d_)
        d_ = self.max2(d_)
        d_ = self.detect3(d_)
        d_ = self.detectbn3(d_)
        d_ = F.relu(d_)
        d_ = self.detect4(d_)
        d_ = self.detectbn4(d_)
        d_ = F.relu(d_)
        d_ = F.avg_pool2d(d_,4)
        d_ = d_.view(d_.size(0),-1)
        d_ = self.detect_l1(d_)
        d_ = F.relu(d_)
        d_ = self.detect_l2(d_)
        #pdb.set_trace()

        if fixed == True:
            out_normal_index = []
            out_adv_index = []
            #pdb.set_trace()
            for i in range(d_.shape[0]):
                if d_.argmax(1)[i] == 0:
                    out_normal_index.append(i)
                elif d_.argmax(1)[i] == 1:
                    out_adv_index.append(i)

                '''
                if d_[i][0] >= d_[i][1]:
                    out_normal_index.append(i)
                else:
                    out_adv_index.append(i)
                '''

            #pdb.set_trace()
            out_normal = out[out_normal_index]
            out_adv = out[out_adv_index]
            for bundle in self.bundles[1:]:
                for block in bundle:
                    out_normal = block(out_normal, 0)
                    out_adv = block(out_adv, 1)
            #pdb.set_trace()
            if 0 in out_normal_index:
                out_final = out_normal[0].unsqueeze(0)
                back_index_normal = 1
                back_index_adv = 0
            elif 0 in out_adv_index:
                out_final = out_adv[0].unsqueeze(0)
                back_index_normal = 0
                back_index_adv = 1

            #pdb.set_trace()
            for i in range(1,d_.shape[0]):
                if i in out_normal_index:
                    out_final = torch.cat((out_final,out_normal[back_index_normal].unsqueeze(0)),0)
                    back_index_normal += 1
                elif i in out_adv_index:
                    out_final = torch.cat((out_final,out_adv[back_index_adv].unsqueeze(0)),0)
                    back_index_adv += 1
            
            #pdb.set_trace()
            out = F.avg_pool2d(out_final, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        else:
            for bundle in self.bundles[1:]:
                for block in bundle:
                    out = block(out, idx2BN)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out, d_

if __name__ == '__main__':
    from thop import profile
    net = ResNet34OAT()
    x = torch.randn(96,3,32,32)
    _lambda = torch.ones(1,1)
    flops, params = profile(net, inputs=(x, _lambda))
    y = net(x, _lambda)
    print(y.size())