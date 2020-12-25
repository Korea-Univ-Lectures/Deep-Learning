import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import importlib
import pickle

#####################################
# author : 송대선 2018320161 컴퓨터학과 #
##################################### 

# put nn_layers and mnist_loader modules to your working directory (or set appropriate paths)
import nn_layers as nnl

class nn_mnist_classifier:
    def __init__(self):
        ## initialize each layer of the entire classifier

        # convolutional layer
        # input image size: 28 x 28
        # filter size: 3 x 3
        # input channel size: 1
        # output channel size (number of filters): 28

        self.conv_layer = nnl.nn_convolutional_layer(Wx_size=2, Wy_size=2, input_size=1,
                                                       in_ch_size=1, out_ch_size=1)
        self.conv_layer.W = np.array([[[[-1, 1], [1, 2]]]])                                               

        # activation layer
        self.act = nnl.nn_activation_layer()


    # forward method
    # parameters:
    #   x: input MNIST images in batch
    #   y: ground truth/labels of the batch
    #   backprop_req: set this to True if backprop method is called next
    #                 set this to False if only forward pass (inference) needed
    def forward(self, x, backprop_req=True):
        ########################
        # Q1. Complete forward method
        ########################
        # cv1_f, ac1_f, mp1_f, fc1_f, ac2_f, fc2_f, sm1_f, cn_f
        # are outputs from each layer of the CNN
        # cv1_f is the output from the convolutional layer
        # ac1_f is the output from 1st activation layer
        # mp1_f is the output from maxpooling layer
        # ... and so on

        #print("x")
        #print(x.shape)
        cv1_f = self.conv_layer.forward(x)
        print("Wx forward", cv1_f)

        # similarly, fill in ... part in the below
        #print("cv1_f")
        #print(cv1_f.shape)
        ac1_f = self.act.forward(cv1_f)
        print("y forward", ac1_f)

        # store intermediate variables, later to be used for backprop
        # store required only when backprop_req is True
        if backprop_req:
            self.fwd_cache = (x, cv1_f, ac1_f)

        # forward will return, scores (sm1_f), and loss (cn_f)
        return ac1_f

    # backprop method
    def backprop(self, dL_dy):
        # note that the backprop will use the saved structures,
        (x, cv1_f, ac1_f) = self.fwd_cache

        ########################
        # Q2. Complete backprop method
        ########################
        #print("y")
        #print(y.shape)

        
        ac_b = self.act.backprop(cv1_f, dL_dy)
        print("dLdl forward", ac_b)

        #print("ac1_b")
        #print(ac1_b.shape)
        cv1_b, dldw_cv1, dldb_cv1 = self.conv_layer.backprop(x, ac_b)

        print("dldw_cv1 forward", dldw_cv1)


        ########################
        # Q2 ends here
        ########################

        # cache upstream gradients for weight updates!
    
        return ac_b, dldw_cv1

if __name__ == "__main__":
    classifier = nn_mnist_classifier()

    x = np.array([[[[1,2,3], [1,1,0], [0, -1, -2]]]])

    loss = classifier.forward(x)

    dL_dy = np.array([[[[-3, 2], [0,1]]]])

    classifier.backprop(dL_dy)
    