import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time

########################################
# You can define whatever classes if needed
########################################

class Block_1(nn.Module):
    def __init__(self, filter_size):
        super(Block_1, self).__init__()

        self.bn_1 = nn.BatchNorm2d(filter_size)
        self.bn_2 = nn.BatchNorm2d(filter_size)

        self.conv_1 = nn.Conv2d(filter_size, filter_size, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(filter_size, filter_size, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        residual = x
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.conv_1(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = x + residual

        return x

class Block_2(nn.Module):
    def __init__(self, filter_size):
        super(Block_2, self).__init__()

        self.bn_1 = nn.BatchNorm2d(filter_size)
        self.bn_2 = nn.BatchNorm2d(filter_size * 2)

        self.conv_res = nn.Conv2d(filter_size, filter_size * 2, kernel_size=1, stride=2, padding=0)
        self.conv_1 = nn.Conv2d(filter_size, filter_size * 2, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(filter_size * 2, filter_size * 2, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = self.bn_1(x)
        x = F.relu(x)
        residual = x
        x = self.conv_1(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.conv_2(x)
        residual = self.conv_res(residual)
        x = x + residual

        return x


class Stage_1(nn.Module):
    def __init__(self, nblk, filter_size):
        super(Stage_1, self).__init__()

        self.nblk = nblk

        self.blocks = list()

        for _ in range(nblk):
            self.blocks.append(Block_1(filter_size))

        self.blocks = nn.Sequential(*self.blocks)


    def forward(self, x):
        if self.nblk > 0:
            x = self.blocks(x)

        return x

class Stage_2(nn.Module):
    def __init__(self, nblk, filter_size):
        super(Stage_2, self).__init__()

        self.nblk = nblk

        self.blocks = list()

        if nblk > 0:
            self.blocks.append(Block_2(filter_size))

        filter_size *= 2

        if nblk > 1:
            for _ in range(nblk - 1):
                self.blocks.append(Block_1(filter_size))

        self.blocks = nn.Sequential(*self.blocks)


    def forward(self, x):
        if self.nblk > 0:
            x = self.blocks(x)

        return x


class IdentityResNet(nn.Module):
    
    # __init__ takes 4 parameters
    # nblk_stage1: number of blocks in stage 1, nblk_stage2.. similar
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()

        self.nblk_stages = (nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4)

        filter_num = 64

        self.input_conv = nn.Conv2d(3, filter_num, kernel_size=3, stride=1, padding=1)

        self.stage_1 = Stage_1(nblk_stage1, filter_num)

        self.stage_2 = Stage_2(nblk_stage1, filter_num)
        filter_num *= 2

        self.stage_3 = Stage_2(nblk_stage1, filter_num)
        filter_num *= 2

        self.stage_4 = Stage_2(nblk_stage1, filter_num)
        filter_num *= 2
       
        self.output_avg_pool = nn.AvgPool2d(kernel_size = 4, stride = 4)
        self.output_fc = nn.Linear(filter_num, 10)

    ########################################
    # Implement the network
    # You can declare whatever variables
    ########################################


    ########################################
    # You can define whatever methods
    ########################################
    
    def forward(self, x):
        ########################################
        # Implement the network
        # You can declare or define whatever variables or methods
        ########################################

        x = self.input_conv(x)

        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)

        x = self.output_avg_pool(x)
        x = x.squeeze()
        out = self.output_fc(x)

        return out

########################################
# Q1. set device
# First, check availability of GPU.
# If available, set dev to "cuda:0";
# otherwise set dev to "cpu"
########################################
dev = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')
print('current device: ', dev)

########################################
# data preparation: CIFAR10
########################################

########################################
# Q2. set batch size
# set batch size for training data
########################################
batch_size = 16

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# define network
net = IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

########################################
# Q3. load model to GPU
# Complete below to load model to GPU
########################################
net = net.to(dev)


# set loss function
criterion = nn.CrossEntropyLoss()

########################################
# Q4. optimizer
# Complete below to use SGD with momentum (alpha= 0.9)
# set proper learning rate
########################################
optimizer = optim.SGD(net.parameters(), lr=0.007, momentum=0.9)

# start training
t_start = time.time()

for epoch in range(5):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)
        
        ########################################
        # Q5. make sure gradients are zero!
        # zero the parameter gradients
        ########################################
        optimizer.zero_grad()
        
        ########################################
        # Q6. perform forward pass
        ########################################
        outputs = net(inputs)
        
        # set loss
        loss = criterion(outputs, labels)
        
        ########################################
        # Q7. perform backprop
        ########################################
        loss.backward()
        
        ########################################
        # Q8. take a SGD step
        ########################################
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end-t_start, ' sec')
            t_start = t_end

print('Finished Training')


# now testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

########################################
# Q9. complete below
# when testing, computation is done without building graphs
########################################
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# per-class accuracy
for i in range(10):
    print('Accuracy of %5s' %(classes[i]), ': ',
          100 * class_correct[i] / class_total[i],'%')

# overall accuracy
print('Overall Accurracy: ', (sum(class_correct)/sum(class_total)), '%')


