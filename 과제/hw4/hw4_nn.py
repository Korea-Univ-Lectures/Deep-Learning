import numpy as np
from skimage.util.shape import view_as_windows

#####################################
# author : 송대선 2018320161 컴퓨터학과 #
#####################################


#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * filter_width * filter_height / 2),
                                  (num_filters, in_ch_size, filter_width, filter_height))
        self.b = 0.01 + np.zeros((1, num_filters, 1, 1))
        self.input_size = input_size

        #######
        ## If necessary, you can define additional class variables here
        #######

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    #######
    # Q1. Complete this method
    #######
    def forward(self, x):
        (batch_size, _, in_width, in_height) = x.shape 
        """print("x")
        print(x)
        print(x.shape)

        print("self.W")
        print(self.W)
        print(self.W.shape)

        print("self.b")
        print(self.b)
        print(self.b.shape)"""

        out = list()
        for x_data in x:
            """print("x_data.shape")
            print(x_data.shape)

            print("self.b[0]")
            print(self.b[0])"""

            activation_layers = list()
            for (W_data, b_data) in zip(self.W, self.b[0]):
                activation_layer = np.zeros((in_width - filter_width + 1, in_height - filter_height + 1))
                #print("activation_layer")
                #print(activation_layer.shape)

                for (x_layer, W_layer) in zip(x_data, W_data):
                    """print("x_layer")
                    print(x_layer.shape)

                    print("W_layer")
                    print(W_layer.shape)"""

                    result = self.conv_layer(x_layer, W_layer, isZeroPadding=False)

                    activation_layer = np.add(activation_layer, result)
                
                bais_matrix = np.ones((in_width - filter_width + 1, in_height - filter_height + 1))*b_data[0,0]
                activation_layer = np.add(activation_layer, bais_matrix)
                activation_layers.append(activation_layer)
            
                #print("activation_layer")
                #print(activation_layer.shape)

            activation_layers = np.array(activation_layers)
            #print("activation_layers")
            #print(activation_layers.shape)

            out.append(activation_layers)
        
        out = np.array(out)
        #print("out")
        #print(out.shape)
            
        #print(out)
        return out

    #######
    # Q2. Complete this method
    #######
    def backprop(self, x, dLdy):
        """print("x", x.shape)
        print("dLdy", dLdy.shape)
        print("W", self.W.shape)
        print("b", self.b.shape)"""

        (batch_size, filter_num, out_w, out_h) = dLdy.shape
        (bacth, in_ch_size, in_w, in_h) = x.shape
        (filter_num, in_ch_size, filt_w, filt_h) = self.W.shape

        dLdx = list()

        for dLdy_batch in dLdy:
            dLdx_batch = np.zeros((in_ch_size, in_w, in_h))

            for (dLdy_filter, W_filter) in zip(dLdy_batch, self.W):
                dLdx_in_ch = list()

                for W_ch in W_filter:
                    #print("dLdy_filter", dLdy_filter.shape)
                    #print("W_ch", W_ch.shape)
                    conv_result = self.conv_layer(dLdy_filter, W_ch, isZeroPadding=True, isReverse=True)
                    #print("conv_result", conv_result.shape)
                    dLdx_in_ch.append(conv_result)

                dLdx_in_ch = np.array(dLdx_in_ch)
                #print("dLdx_in_ch", dLdx_in_ch.shape)
                dLdx_batch += dLdx_in_ch

            dLdx.append(dLdx_batch)

        dLdx = np.array(dLdx)
        #print("dLdx", dLdx.shape)


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

        #print("dLdW", dLdW.shape)

        dLdb = dLdy
        dLdb = dLdb.sum(axis=3)
        dLdb = dLdb.sum(axis=2)
        dLdb = dLdb.sum(axis=0)
        dLdb = dLdb.reshape(self.b.shape)

        #print("dLdb", dLdb.shape)

        return dLdx, dLdW, dLdb

    #######
    ## If necessary, you can define additional class methods here
    #######

    # This methods compute conv one chennel by chennel.
    # You can also determine whether use zero padding or not. 
    def conv_layer(self, input_layer, filter_layer, isZeroPadding=False, isReverse=False):
        #print(input_layer.shape)
        #print(filter_layer.shape)

        
        (filter_width, filter_height) = filter_layer.shape

        if isZeroPadding:
            input_layer = np.pad(input_layer, ((filter_height-1,filter_height-1),(filter_width-1,filter_width-1)), 'constant', constant_values=0)

        if isReverse:
            filter_layer = np.flip(filter_layer, axis=0)
            filter_layer = np.flip(filter_layer, axis=1)

        (in_width, in_height) = input_layer.shape
        
        #print("input_layer", input_layer.shape)

        y = view_as_windows(input_layer, filter_layer.shape)
        #print("y")
        #print(y.shape)



        y = y.reshape((in_width - filter_width + 1, in_height - filter_height + 1,-1))
        #print("y")
        #print(y.shape)

                    
        #print("reshape W_layer")
        #print(W_layer.reshape((-1,1)))

        result = y.dot(filter_layer.reshape((-1,1)))

        result = np.squeeze(result,axis=2)

        return result


class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        #######
        ## If necessary, you can define additional class variables here
        #######

    #######
    # Q3. Complete this method
    #######
    def forward(self, x):
        #print("x", x.shape)

        (_, _, x_w, x_h) = x.shape
        pool_window = (self.pool_size, self.pool_size)

        out = list()
        for x_batch in x:
            out_batch = list()
            for x_in_ch in x_batch:
                y = view_as_windows(x_in_ch, pool_window, step=self.stride)
                #print("y")
                #print(y.shape)
                #print(y)

                y = y.reshape((int((x_w-self.pool_size)/self.stride) + 1, int((x_h-self.pool_size)/self.stride) + 1,-1))
                #print("y")
                #print(y.shape)

                            
                #print("reshape W_layer")
                #print(W_layer.reshape((-1,1)))

                result = y.max(axis=2)
                out_batch.append(result)

            out_batch = np.array(out_batch)
            out.append(out_batch)

        out = np.array(out)

        #print("x", x.shape)
        #print("out", out.shape)

        return out

    #######
    # Q4. Complete this method
    #######
    def backprop(self, x, dLdy):
        #print("x", x.shape)
        #print(x)
        #print("dLdy", dLdy.shape)
        #print(dLdy)

        pool_window = (self.pool_size, self.pool_size)

        (batch_size, in_ch, in_w, in_h) = x.shape

        #dLdx = x
        #print(dLdx)
        #dLdx = dLdx.reshape((batch_size, in_ch, -1))
        #print(dLdx)
 
        dLdx = list()
        for (x_batch, dLdy_batch) in zip(x, dLdy):
            dLdx_batch = list()
            for (x_in_ch, dLdy_in_ch) in zip(x_batch, dLdy_batch):

                dLdx_in_ch = np.zeros(x_in_ch.shape)

                y = view_as_windows(x_in_ch, pool_window, step=self.stride)
                #print("y", y.shape)

                for width_number, (y_width, dLdy_width) in enumerate(zip(y, dLdy_in_ch)):
                    for height_number, (y_layer, dLdy_value) in enumerate(zip(y_width, dLdy_width)):
                        (y_layer_w, y_layer_h) = y_layer.shape
                        y_max_index = np.argmax(y_layer)
                        y_max_index_width = y_max_index % y_layer_w
                        y_max_index_height = int(y_max_index / y_layer_w)

                        #print(width_number*self.stride + y_max_index_width)
                        #print(height_number*self.stride + y_max_index_height)
                        dLdx_in_ch[width_number*self.stride + y_max_index_width, height_number*self.stride + y_max_index_height] = dLdy_value
                
                dLdx_batch.append(dLdx_in_ch)

            dLdx_batch = np.array(dLdx_batch)
            dLdx.append(dLdx_batch)
        dLdx = np.array(dLdx)




        """dLdx = dLdx.reshape((batch_size, in_ch, -1))

        x = x.reshape((batch_size, in_ch, -1))
        x = (x.max(axis=2, keepdims=True) == x)
        x = 1*x
        #print(x)

        dLdy = dLdy.reshape((batch_size, in_ch, -1))
        print("dLdy", dLdy.shape)"""

        
        return dLdx

    #######
    ## If necessary, you can define additional class methods here
    #######


# testing the implementation

# data sizes
batch_size = 8
input_size = 32
filter_width = 3
filter_height = filter_width
in_ch_size = 3
num_filters = 8

std = 1e0
dt = 1e-3

# number of test loops
num_test = 20

# error parameters
err_dLdb = 0
err_dLdx = 0
err_dLdW = 0
err_dLdx_pool = 0

for i in range(num_test):
    # create convolutional layer object
    cnv = nn_convolutional_layer(filter_width, filter_height, input_size, in_ch_size, num_filters, std)

    x = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
    #print(x)
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    # dLdx test
    print('dLdx test')
    y1 = cnv.forward(x)
    y2 = cnv.forward(x + delta)

    bp, _, _ = cnv.backprop(x, np.ones(y1.shape))

    exact_dx = np.sum(y2 - y1) / dt
    apprx_dx = np.sum(delta * bp) / dt
    print('exact change', exact_dx)
    print('apprx change', apprx_dx)

    err_dLdx += abs((apprx_dx - exact_dx) / exact_dx) / num_test * 100

    # dLdW test
    print('dLdW test')
    W, b = cnv.get_weights()
    dW = np.random.normal(0, 1, W.shape) * dt
    db = np.zeros(b.shape)

    z1 = cnv.forward(x)
    #print("s1.shape")
    #print(z1.shape)
    _, bpw, _ = cnv.backprop(x, np.ones(z1.shape))
    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_dW = np.sum(z2 - z1) / dt
    apprx_dW = np.sum(dW * bpw) / dt
    print('exact change', exact_dW)
    print('apprx change', apprx_dW)

    err_dLdW += abs((apprx_dW - exact_dW) / exact_dW) / num_test * 100

    # dLdb test
    print('dLdb test')

    W, b = cnv.get_weights()

    dW = np.zeros(W.shape)
    db = np.random.normal(0, 1, b.shape) * dt

    z1 = cnv.forward(x)

    V = np.random.normal(0, 1, z1.shape)

    _, _, bpb = cnv.backprop(x, V)

    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_db = np.sum(V * (z2 - z1) / dt)
    apprx_db = np.sum(db * bpb) / dt

    print('exact change', exact_db)
    print('apprx change', apprx_db)
    err_dLdb += abs((apprx_db - exact_db) / exact_db) / num_test * 100

    # max pooling test
    # parameters for max pooling
    stride = 2
    pool_size = 2

    mpl = nn_max_pooling_layer(stride=stride, pool_size=pool_size)

    x = np.arange(batch_size * in_ch_size * input_size * input_size).reshape(
        (batch_size, in_ch_size, input_size, input_size)) + 1
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    print('dLdx test for pooling')
    y1 = mpl.forward(x)
    dLdy = np.random.normal(0, 10, y1.shape)
    bpm = mpl.backprop(x, dLdy)

    y2 = mpl.forward(x + delta)

    exact_dx_pool = np.sum(dLdy * (y2 - y1)) / dt
    apprx_dx_pool = np.sum(delta * bpm) / dt
    print('exact change', exact_dx_pool)
    print('apprx change', apprx_dx_pool)

    err_dLdx_pool += abs((apprx_dx_pool - exact_dx_pool) / exact_dx_pool) / num_test * 100

# reporting accuracy results.
print('accuracy results')
print('conv layer dLdx', 100 - err_dLdx, '%')
print('conv layer dLdW', 100 - err_dLdW, '%')
print('conv layer dLdb', 100 - err_dLdb, '%')
print('maxpool layer dLdx', 100 - err_dLdx_pool, '%')