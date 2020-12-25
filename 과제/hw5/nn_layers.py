import numpy as np
from skimage.util.shape import view_as_windows

#####################################
# author : 송대선 2018320161 컴퓨터학과 #
##################################### 

##########
#   convolutional layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_convolutional_layer:

    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):
    
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        (batch_size, _, in_width, in_height) = x.shape 
        (num_filters, in_ch_size, filter_width, filter_height) = self.W.shape

        out = list()
        for x_data in x:

            activation_layers = list()
            for (W_data, b_data) in zip(self.W, self.b[0]):
                activation_layer = np.zeros((in_width - filter_width + 1, in_height - filter_height + 1))

                for (x_layer, W_layer) in zip(x_data, W_data):


                    result = self.conv_layer(x_layer, W_layer, isZeroPadding=False)

                    activation_layer = np.add(activation_layer, result)
                
                bais_matrix = np.ones((in_width - filter_width + 1, in_height - filter_height + 1))*b_data[0,0]
                activation_layer = np.add(activation_layer, bais_matrix)
                activation_layers.append(activation_layer)
            
            activation_layers = np.array(activation_layers)

            out.append(activation_layers)
        
        out = np.array(out)

        return out

    def backprop(self, x, dLdy):

        (batch_size, filter_num, out_w, out_h) = dLdy.shape
        (bacth, in_ch_size, in_w, in_h) = x.shape
        (filter_num, in_ch_size, filt_w, filt_h) = self.W.shape

        dLdx = list()

        for dLdy_batch in dLdy:
            dLdx_batch = np.zeros((in_ch_size, in_w, in_h))

            for (dLdy_filter, W_filter) in zip(dLdy_batch, self.W):
                dLdx_in_ch = list()

                for W_ch in W_filter:

                    conv_result = self.conv_layer(dLdy_filter, W_ch, isZeroPadding=True, isReverse=True)
                    dLdx_in_ch.append(conv_result)

                dLdx_in_ch = np.array(dLdx_in_ch)
                dLdx_batch += dLdx_in_ch

            dLdx.append(dLdx_batch)

        dLdx = np.array(dLdx)


        dLdW = np.zeros(self.W.shape)

        for (dLdy_batch, x_batch) in zip(dLdy, x):
            dLdW_filter = list()
            for dLdy_filter in dLdy_batch:
                dLdW_in_ch = list()
                for x_in_ch in x_batch:
                    conv_result = self.conv_layer(x_in_ch, dLdy_filter)
                    dLdW_in_ch.append(conv_result)

                dLdW_in_ch = np.array(dLdW_in_ch)
                dLdW_filter.append(dLdW_in_ch)
            
            dLdW_filter = np.array(dLdW_filter)
            dLdW = dLdW + dLdW_filter

        dLdb = dLdy
        dLdb = dLdb.sum(axis=3)
        dLdb = dLdb.sum(axis=2)
        dLdb = dLdb.sum(axis=0)
        dLdb = dLdb.reshape(self.b.shape)


        return dLdx, dLdW, dLdb

    #######
    ## If necessary, you can define additional class methods here
    #######

    def conv_layer(self, input_layer, filter_layer, isZeroPadding=False, isReverse=False):
        (filter_width, filter_height) = filter_layer.shape

        if isZeroPadding:
            input_layer = np.pad(input_layer, ((filter_height-1,filter_height-1),(filter_width-1,filter_width-1)), 'constant', constant_values=0)

        if isReverse:
            filter_layer = np.flip(filter_layer, axis=0)
            filter_layer = np.flip(filter_layer, axis=1)

        (in_width, in_height) = input_layer.shape
        
        y = view_as_windows(input_layer, filter_layer.shape)

        y = y.reshape((in_width - filter_width + 1, in_height - filter_height + 1,-1))

        result = y.dot(filter_layer.reshape((-1,1)))

        result = np.squeeze(result,axis=2)

        return result


##########
#   max pooling layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):

        (_, _, x_w, x_h) = x.shape
        pool_window = (self.pool_size, self.pool_size)

        out = list()
        for x_batch in x:
            out_batch = list()
            for x_in_ch in x_batch:
                y = view_as_windows(x_in_ch, pool_window, step=self.stride)
                y = y.reshape((int((x_w-self.pool_size)/self.stride) + 1, int((x_h-self.pool_size)/self.stride) + 1,-1))

                result = y.max(axis=2)
                out_batch.append(result)

            out_batch = np.array(out_batch)
            out.append(out_batch)

        out = np.array(out)

        return out

    def backprop(self, x, dLdy):

        pool_window = (self.pool_size, self.pool_size)

        (batch_size, in_ch, in_w, in_h) = x.shape
 
        dLdx = list()
        for (x_batch, dLdy_batch) in zip(x, dLdy):
            dLdx_batch = list()
            for (x_in_ch, dLdy_in_ch) in zip(x_batch, dLdy_batch):

                dLdx_in_ch = np.zeros(x_in_ch.shape)

                y = view_as_windows(x_in_ch, pool_window, step=self.stride)

                for width_number, (y_width, dLdy_width) in enumerate(zip(y, dLdy_in_ch)):
                    for height_number, (y_layer, dLdy_value) in enumerate(zip(y_width, dLdy_width)):
                        (y_layer_w, y_layer_h) = y_layer.shape
                        y_max_index = np.argmax(y_layer)
                        y_max_index_width = y_max_index % y_layer_w
                        y_max_index_height = int(y_max_index / y_layer_w)
                        dLdx_in_ch[width_number*self.stride + y_max_index_width, height_number*self.stride + y_max_index_height] = dLdy_value
                
                dLdx_batch.append(dLdx_in_ch)

            dLdx_batch = np.array(dLdx_batch)
            dLdx.append(dLdx_batch)
        dLdx = np.array(dLdx)

        return dLdx



##########
#   fully connected layer
##########
# fully connected linear layer.
# parameters: weight matrix matrix W and bias b
# forward computation of y=Wx+b
# for (input_size)-dimensional input vector, outputs (output_size)-dimensional vector
# x can come in batches, so the shape of y is (batch_size, output_size)
# W has shape (output_size, input_size), and b has shape (output_size,)

class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        # Xavier/He init
        self.W = np.random.normal(0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+np.zeros((output_size))

    def forward(self,x):
        out = list()

        
        #print("reshaped x")
        #print(x.shape)

        #print("W")
        #print(self.W.shape)

        for data_x in x:
            #print("one x")
            #print(data_x.shape)

            xW = data_x @ self.W.T
            #print("xW")
            #print(xW.shape)
            #print(xW.ndim)

            #data_y = xW + np.squeeze(self.b.T, axis=0)
            out.append(xW)

        out = np.array(out)
        #print("y")
        #print(y)
        return out
    
    def backprop(self,x,dLdy):

        dLdx = list()
        dLdW = np.zeros(self.W.shape)
        dLdb = np.zeros(self.b.shape).T

        for (data_x, data_dLdy) in zip(x, dLdy):

            data_dLdx = data_dLdy @ self.W

            dLdx.append(data_dLdx)

            data_dLdW = np.outer(data_dLdy, data_x)

            dLdW += data_dLdW

            data_dLdb = data_dLdy

            dLdb += data_dLdb


        dLdx = np.array(dLdx)

        n = x.shape[0]
        dLdW = dLdW / n

        dLdb = dLdb / n

        return dLdx,dLdW,dLdb

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

##########
#   activation layer
##########
#   This is ReLU activation layer.
##########

class nn_activation_layer:
    
    def __init__(self):
        self.chk = None
 
    def forward(self, x):
        self.chk = (x>0).astype(np.int)
        return x * self.chk
 
    def backprop(self, x, dout):
        return dout * self.chk


##########
#   softmax layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_softmax_layer:

    def __init__(self):
        pass

    def forward(self,x):
        y = list()

        for data_x in x:

            exp_x = np.exp(data_x)
            sum_exp_x = np.sum(exp_x)
            data_y = exp_x/sum_exp_x 
            y.append(data_y)

        y = np.array(y)
        return y
    

    def backprop(self,x,dLdy):

        s = self.forward(x)

        dLdx = list()
        for (data_s, data_dLdy) in zip(s, dLdy):
            data_dsdx = np.outer(-data_s, data_s)
            diag_s = np.diag(data_s)
            data_dsdx = data_dsdx + diag_s

            #print("data_dLdy")
            #print(data_dLdy.shape)

            #print("data_dsdx")
            #print(data_dsdx.shape)

            data_dLdx = data_dsdx @ data_dLdy
            dLdx.append(data_dLdx)

        dLdx = np.array(dLdx)
        return dLdx


##########
#   cross entropy layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_cross_entropy_layer:

    def __init__(self):
        pass

    def forward(self,x,y):
        L = 0
        for (data_x, data_y) in zip(x,y):

            L -= np.log(data_x[data_y])

        L /= x.shape[0]

        return L
        
    def backprop(self,x,y):
        dLdx = list()

        for (data_x, data_y) in zip(x,y):

            data_dLdx = np.zeros((data_x.shape[0], ))
            #if data_x[data_y] != 0:
            data_dLdx[data_y] = -1/data_x[data_y]

            dLdx.append(data_dLdx)
        
        dLdx = np.array(dLdx)

        return dLdx

