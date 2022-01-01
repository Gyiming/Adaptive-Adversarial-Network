import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class ptomlp(nn.Module):

    def __init__(self):
        '''
        Args:
            num_features: int. Number of channels
        '''
        super(ptomlp, self).__init__()
        self.l1 = nn.Linear(18, 128)
        torch.nn.init.xavier_uniform(self.l1.weight)
        self.l1.bias.data.fill_(0.01)
        self.l2 = nn.Linear(128, 256)
        torch.nn.init.xavier_uniform(self.l2.weight)
        self.l2.bias.data.fill_(0.01)
        self.l3 = nn.Linear(256,3)
        torch.nn.init.xavier_uniform(self.l3.weight)
        self.l3.bias.data.fill_(0.01)
        #self.BN_a2 = nn.BatchNorm2d(num_features)

    def forward(self, x):
        out = self.l1(x)
        out = F.relu(out)
        out = self.l2(out)
        out = F.relu(out)
        out = self.l3(out)

        return out





    