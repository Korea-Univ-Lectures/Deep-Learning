import sys
import os
import matplotlib.pyplot as plt
import numpy as np

class nn_linear_layer:
    
    # linear layer.
    # randomly initialized by creating matrix W and bias b
    def __init__(self, input_size, output_size, std=1):
        self.W = np.random.normal(0,std,(output_size,input_size))
        self.b = np.random.normal(0,std,(output_size,1))
    
    ######
    ## Q1
    def forward(self,x):
        y = list()

        for data_x in x:
            """print("data_x")
            print(data_x)
            print("self.W")
            print(self.W)
            print("self.b")
            print(self.b)"""

            xW = data_x @ self.W.T
            #print("xW")
            #print(xW)
            
            data_y = xW + np.squeeze(self.b.T, axis=0)
            #print("data_y")
            #print(data_y)
            y.append(data_y)

        y = np.array(y)
        #print("y")
        #print(y)
        return y
    
    ######
    ## Q2
    ## returns three parameters
    def backprop(self,x,dLdy):
        #print("x")
        #print(x)
        #print("dLdy")
        #print(dLdy)

        #print("self.W")
        #print(self.W)
        #print("self.b")
        #print(self.b)

        dLdx = list()
        dLdW = np.zeros(self.W.shape)
        dLdb = np.zeros(self.b.shape).T

        for (data_x, data_dLdy) in zip(x, dLdy):
            #print("data_x")
            #print(data_x)

            #print("data_dLdy")
            #print(data_dLdy)

            data_dLdx = data_dLdy @ self.W
            #print("data_dLdx")
            #print(data_dLdx)

            dLdx.append(data_dLdx)

            data_dLdW = np.outer(data_dLdy, data_x)
            #print("data_dLdW")
            #print(data_dLdW)

            dLdW += data_dLdW

            data_dLdb = data_dLdy
            #print("data_dLdb")
            #print(data_dLdb)

            dLdb += data_dLdb


        dLdx = np.array(dLdx)
        #print("dLdx")
        #print(dLdx)

        n = x.shape[0]
        dLdW = dLdW / n
        #print("dLdW")
        #print(dLdW)

        dLdb = dLdb / n
        #print("dLdb")
        #print(dLdb)

        return dLdW,dLdb,dLdx

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

class nn_activation_layer:
    
    def __init__(self):
        pass
    
    ######
    ## Q3
    def forward(self,x):
        y = 1/(1+np.exp(-x))
        #print("y")
        #print(y)
        return y
    
    ######
    ## Q4
    def backprop(self,x,dLdy):
        #print("x")
        #print(x)

        #print("dLdy")
        #print(dLdy)

        s = self.forward(x)
        #print("s")
        #print(s)
        
        dLdx = dLdy*(1-s)*s

        #print("dLdx")
        #print(dLdx)

        return dLdx


class nn_softmax_layer:
    def __init__(self):
        pass
    ######
    ## Q5
    def forward(self,x):
        y = list()

        for data_x in x:
            #print("data_x")
            #print(data_x)

            exp_x = np.exp(data_x)
            #print("exp_x")
            #print(exp_x)

            #print("sum_exp_x")
            sum_exp_x = np.sum(exp_x)
            #print(sum_exp_x)

            #print("data_y")
            data_y = exp_x/sum_exp_x 
            #print(data_y)
            
            #print("data_y")
            #print(data_y)
            y.append(data_y)

        y = np.array(y)
        #print("y")
        #print(y)
        return y
    
    ######
    ## Q6
    def backprop(self,x,dLdy):
        #print("x")
        #print(x)

        #print("dLdy")
        #print(dLdy)

        #print("s")
        s = self.forward(x)
        #print(s)

        dLdx = list()
        for (data_s, data_dLdy) in zip(s, dLdy):
            data_dsdx = np.outer(-data_s, data_s)
            #print("data_dsdx")
            #print(data_dsdx)

            diag_s = np.diag(data_s)
            #print("diag_s")
            #print(diag_s)

            data_dsdx = data_dsdx + diag_s
            #print("data_dsdx")
            #print(data_dsdx)

            #print("data_dLdy")
            #print(data_dLdy)

            data_dLdx = data_dLdy @ data_dsdx
            #print("data_dLdx")
            #print(data_dLdx)

            dLdx.append(data_dLdx)

        dLdx = np.array(dLdx)
        #print("dLdx")
        #print(dLdx)
        return dLdx

class nn_cross_entropy_layer:
    def __init__(self):
        pass
        
    ######
    ## Q7
    def forward(self,x,y):
        L = 0
        for (data_x, data_y) in zip(x,y):
            #print("data_x")
            #print(data_x)

            #print("data_y")
            #print(data_y)
            L -= (1-data_y)*np.log(data_x[0]) + data_y*np.log(data_x[1])

        L /= x.shape[0]

        return L
        
    ######
    ## Q8
    def backprop(self,x,y):
        dLdx = list()

        for (data_x, data_y) in zip(x,y):
            #print("data_x")
            #print(data_x)

            #print("data_y")
            #print(data_y)

            data_dLdx = np.zeros((2, ))
            #print("data_dLdx")
            #print(data_dLdx)
            if data_y == 0:
                data_dLdx[0] = -1/data_x[0]

            if data_y == 1:
                data_dLdx[1] = -1/data_x[1]
            
            #print("data_dLdx")
            #print(data_dLdx)

            dLdx.append(data_dLdx)
        
        dLdx = np.array(dLdx)
        #print("dLdx")
        #print(dLdx)

        return dLdx

# number of data points for each of (0,0), (0,1), (1,0) and (1,1)
num_d=5

# number of test runs
num_test=40

## Q9. Hyperparameter setting
## learning rate (lr)and number of gradient descent steps (num_gd_step)
## This part is not graded (there is no definitive answer).
## You can set this hyperparameters through experiments.
lr=0.07
num_gd_step=4000

# dataset size
batch_size=4*num_d

# number of classes is 2
num_class=2

# variable to measure accuracy
accuracy=0

# set this True if want to plot training data
show_train_data=True

# set this True if want to plot loss over gradient descent iteration
show_loss=True

for j in range(num_test):
    
    # create training data
    m_d1=(0,0)
    m_d2=(1,1)
    m_d3=(0,1)
    m_d4=(1,0)

    sig=0.05
    s_d1=sig**2*np.eye(2)

    d1=np.random.multivariate_normal(m_d1,s_d1,num_d)
    d2=np.random.multivariate_normal(m_d2,s_d1,num_d)
    d3=np.random.multivariate_normal(m_d3,s_d1,num_d)
    d4=np.random.multivariate_normal(m_d4,s_d1,num_d)

    # training data, and has dimension (4*num_d,2,1)
    x_train_d = np.vstack((d1,d2,d3,d4))
    # training data lables, and has dimension (4*num_d,1)
    y_train_d = np.vstack((np.zeros((2*num_d,1),dtype='uint8'),np.ones((2*num_d,1),dtype='uint8')))
    
    # plotting training data if needed
    if (show_train_data) & (j==0):
        plt.grid()
        plt.scatter(x_train_d[range(2*num_d),0], x_train_d[range(2*num_d),1], color='b', marker='o')
        plt.scatter(x_train_d[range(2*num_d,4*num_d),0], x_train_d[range(2*num_d,4*num_d),1], color='r', marker='x')
        plt.show()
                        
    # create layers

    # hidden layer
    # linear layer
    layer1= nn_linear_layer(input_size=2,output_size=4,)
    # activation layer
    act=nn_activation_layer()
                            
                            
    # output layer
    # linear
    layer2= nn_linear_layer(input_size=4,output_size=2,)
    # softmax
    smax=nn_softmax_layer()
    # cross entropy
    cent=nn_cross_entropy_layer()


    # variable for plotting loss
    loss_out=np.zeros((num_gd_step))

    for i in range(num_gd_step):
        
        # fetch data
        x_train = x_train_d
        y_train = y_train_d
            
        # create one-hot vectors from the ground truth labels
        y_onehot = np.zeros((batch_size,num_class))
        y_onehot[range(batch_size),y_train.reshape(batch_size,)]=1

        ################
        # forward pass
        
        # hidden layer
        # linear
        l1_out=layer1.forward(x_train)
        # activation
        a1_out=act.forward(l1_out)
    
        # output layer
        # linear
        l2_out=layer2.forward(a1_out)
        # softmax
        smax_out=smax.forward(l2_out)
        # cross entropy loss
        loss_out[i]=cent.forward(smax_out,y_train)
            
        ################
        # perform backprop
        # output layer
        
        # cross entropy
        b_cent_out=cent.backprop(smax_out,y_train)
        # softmax
        b_nce_smax_out=smax.backprop(l2_out,b_cent_out)
            
        # linear
        b_dLdW_2,b_dLdb_2,b_dLdx_2=layer2.backprop(x=a1_out,dLdy=b_nce_smax_out)
    
        # backprop, hidden layer
        # activation
        b_act_out=act.backprop(x=l1_out,dLdy=b_dLdx_2)
        # linear
        b_dLdW_1,b_dLdb_1,b_dLdx_1=layer1.backprop(x=x_train,dLdy=b_act_out)
    
        ################
        # update weights: perform gradient descent
        layer2.update_weights(dLdW=-b_dLdW_2*lr,dLdb=-b_dLdb_2.T*lr)
        layer1.update_weights(dLdW=-b_dLdW_1*lr,dLdb=-b_dLdb_1.T*lr)

        if (i+1) % 2000 ==0:
            print('gradient descent iteration:',i+1)
                
    # set show_loss to True to plot the loss over gradient descent iterations
    if (show_loss) & (j==0):
        plt.figure(1)
        plt.grid()
        plt.plot(range(num_gd_step),loss_out)
        plt.xlabel('number of gradient descent steps')
        plt.ylabel('cross entropy loss')
        plt.show()
    
    
    ################
    # training done
    # now testing

    predicted=np.ones((4,))

    # predicting label for (1,1)
    l1_out=layer1.forward([[1,1]])
    a1_out=act.forward(l1_out)
    l2_out=layer2.forward(a1_out)
    smax_out=smax.forward(l2_out)
    predicted[0] = np.argmax(smax_out)
    print('softmax out for (1,1)',smax_out,'predicted label:',int(predicted[0]))

    # predicting label for (0,0)
    l1_out=layer1.forward([[0,0]])
    a1_out=act.forward(l1_out)
    l2_out=layer2.forward(a1_out)
    smax_out=smax.forward(l2_out)
    predicted[1] = np.argmax(smax_out)
    print('softmax out for (0,0)',smax_out,'predicted label:',int(predicted[1]))

    # predicting label for (1,0)
    l1_out=layer1.forward([[1,0]])
    a1_out=act.forward(l1_out)
    l2_out=layer2.forward(a1_out)
    smax_out=smax.forward(l2_out)
    predicted[2] = np.argmax(smax_out)
    print('softmax out for (1,0)',smax_out,'predicted label:',int(predicted[2]))

    # predicting label for (0,1)
    l1_out=layer1.forward([[0,1]])
    a1_out=act.forward(l1_out)
    l2_out=layer2.forward(a1_out)
    smax_out=smax.forward(l2_out)
    predicted[3] = np.argmax(smax_out)
    print('softmax out for (0,1)',smax_out,'predicted label:',int(predicted[3]))

    print('total predicted labels:',predicted.astype('uint8'))
        
    accuracy += (predicted[0]==0)&(predicted[1]==0)&(predicted[2]==1)&(predicted[3]==1)

    if (j+1)%10==0:
        print('test iteration:',j+1)

print('accuracy:',accuracy/num_test*100,'%')






