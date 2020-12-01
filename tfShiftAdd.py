
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K

def matrix_multiply(m,n):
        count = 0
        l = m.shape[0]
        # l=28
        temp = np.zeros((l, l))

        for i in range(l):
            for j in range(l):
             
                count = 0
                m_ = m[i][j]
                n_ = n[i][j]

                while (m_):
                    if (m_ % 2 == 1): 
                        temp[i][j] += n_ << count 
                    count += 1
                    m_ = int(m_/2) 
        return temp
a_function = tf.function(matrix_multiply)



class MyLayer(tf.keras.layers.Layer):

    def __init__(self, n_f, kernel_size, num_output):
        super(MyLayer, self).__init__()

        self.n_f = n_f
        # self.n_c n_c
        self.kernel_size = kernel_size
        self.num_output = num_output


    def build(self, input_shape):
        _, _,self.in_dim_,self.n_c = input_shape

        self.w = self.add_weight(shape=(self.in_dim_,self.in_dim_,self.num_output),
                               initializer="random_normal",
                               trainable=True,)
        
        super(MyLayer, self).build(input_shape)



    def matrix_multiply(self, m,n):
        count = 0
        l = m.shape[0]
        temp = tf.zeros((l, l))
        for i in range(l):
            for j in range(l):
             
                count = 0
                m_ = m[i][j]
                n_ = n[i][j]

                while (m_):
                    if (m_ % 2 == 1): 
                        temp[i][j] += n_ << count 
                    count += 1
                    m_ = int(m_/2) 
        return temp


    def call(self, inputs):
        image = inputs[0]
        
        # image = inputs[:,1:]
        in_dim = self.in_dim_
        # image = tf.reshape(in_dim,in_dim, 1)
        f = self.in_dim_
        s=1
        
        # n_f = self.n_f
        n_f = self.num_output
        out_dim = int((in_dim - f)/s)+1 # calculate output dimensions

        out = K.zeros((1,out_dim, out_dim, n_f), dtype='float64')

        for curr_f in range(n_f):
            curr_y = out_y = 0
            while curr_y + f <= in_dim:
                curr_x = out_x = 0
                while curr_x + f <= in_dim:
                
                    filt = self.w
                    filt = filt[:,:, curr_f] 
                    a = a_function(filt, image[curr_y:curr_y+f, curr_x:curr_x+f, curr_f])
                    out = out[0, out_y, out_x, curr_f].assign(tf.math.reduce_sum(a))
                    

                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
        
        keras_shape = keras.backend.int_shape(inputs) 
        tf_shape_tuple = (-1 , 1, 1, 1)
        return tf.reshape(out , tf_shape_tuple)
        
        
    def compute_output_shape(self, input_shape):
        return (-1, input_shape[0],input_shape[1], input_shape[2])
