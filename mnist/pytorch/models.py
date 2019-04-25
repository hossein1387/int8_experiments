# -*- coding: utf-8 -*-
import numpy            as np
from   torch.nn     import (Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d, ReLU, BatchNorm1d, Linear,
                            CrossEntropyLoss)
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict 

#
# For the sake of ease of use, all models below inherit from this class, which
# exposes a constrain() method that allows re-applying the module's constraints
# to its parameters.
#
class BaseModel(torch.nn.Module):
    def constrain(self):
        def fn(module):
            if module is not self and hasattr(module, "constrain"):
                module.constrain()
        self.apply(fn)
    # custom weights initialization called on netG and netD
    def weights_init(self, config):
        classname = self.__class__.__name__
        if classname.find("Linear") != -1:
            config = self.config
            if config['init_type']== "zero":
                self.weight.data.fill_(0)
                self.bias.data.fill_(0)
            elif config['init_type'] == "normal":
                self.weight.data.normal_(0, 1)
                self.bias.data.fill_(0)
            elif config['init_type'] == "glorot":
                f_in  = np.shape(self.weight)[1]
                f_out = np.shape(self.weight)[0]
                glorot_init = np.sqrt(6.0/(f_out+f_in))
                self.weight.data.uniform_(-glorot_init, glorot_init)
                self.bias.data.fill_(0)
            else:
                print ("Unsupported config type".format(config['init_type']))
                sys.exit()

class LENET(BaseModel):
    def __init__(self, config):
        super(LENET, self).__init__()
        self.use_batch_norm = True
        self.passcnt   = 0
        self.config    = config
        self.cnn1      = Conv2d(1,   6  , kernel_size=(5, 5), padding=2)
        self.cnn2      = Conv2d(6,   16 , kernel_size=(5, 5), padding=2)
        self.cnn3      = Conv2d(16,  120, kernel_size=(5, 5), padding=2)

        self.fc1       = Linear(7*7*120, 120)
        self.fc2       = Linear(120, 84)
        self.fc3       = Linear(84, 10)

        self.conv1b_bn = BatchNorm2d(6   )
        self.conv2b_bn = BatchNorm2d(16  )
        self.conv3b_bn = BatchNorm2d(120 )
        # self.conv4b_bn = BatchNorm1d(120 )
        # self.conv5b_bn = BatchNorm1d(84  )
        # self.conv6b_bn = BatchNorm1d(10  )

        self.maxpool   = MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        #import ipdb as pdb; pdb.set_trace()
        if(self.use_batch_norm):
            #layer1
            out = self.cnn1(x)
            out = self.conv1b_bn(out)
            out = self.maxpool(out)
            #layer2
            out = self.cnn2(out)
            out = self.conv2b_bn(out)
            out = self.maxpool(out)
            #layer3
            out = self.cnn3(out)
            out = self.conv3b_bn(out)

            out = out.view(out.size(0), -1)

            out = self.fc1(out)
            # out = self.conv4b_bn(out)
            out = self.fc2(out)
            # out = self.conv5b_bn(out)
            out = self.fc3(out)
            # out = self.conv6b_bn(out)
        else:
            layer1_out = self.maxpool(self.cnn1(x))
            layer2_out = self.maxpool(self.cnn2(layer1_out))
            layer3_out = self.cnn3(layer2_out)
            out        = layer3_out.view(layer3_out.size(0), -1)
            # print("max={0}, min={1}".format(out.max(), out.min()))
            out        = self.fc1(out)
            out        = self.fc2(out)
            out        = self.fc3(out)
        return out

    def _activation(self, x):
        x = F.relu(x)
        return x


# building model
class MLP(BaseModel):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(self.config['batchsize'], 28*28)
        out = self._activation(self.fc1(x))
        out = self._activation(self.fc2(out))
        out = self.fc3(out)
        return out
    def _activation(self, x):
        x = F.relu(x)
        return x
