import tensorflow as tf
class MyLayer(tf.keras.layers.Layer):

    def __init__(self, n_f, kernel_size, num_output):
        super(MyLayer, self).__init__()

        self.n_f = n_f
        # self.n_c n_c
        self.kernel_size = kernel_size
        self.num_output = num_output


    def build(self, input_shape):
        self.in_dim_, _,self.n_c = input_shape

        self.w = self.add_weight(shape=(input_shape[0], input_shape[0],self.num_output),
                               initializer="random_normal",
                               trainable=True,)




    

    def matrix_multiply(m,n):
        count = 0
        l = n.shape[0]
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
        image = inputs
        in_dim = self.in_dim_
        f = self.in_dim_
        s=1
        n_f = self.n_f
        out_dim = int((in_dim - f)/s)+1 # calculate output dimensions
        # bias = 0.001*np.ones((n_f, 1))

        out = np.zeros((out_dim,out_dim, n_f))
        
        for curr_f in range(n_f):
            curr_y = out_y = 0
        # move filter vertically across the image
            while curr_y + f <= in_dim:
                curr_x = out_x = 0
            # move filter horizontally across the image 
                while curr_x + f <= in_dim:
                
        #         # state of art is here:
                    filt = self.w
                    filt = filt[:,:, curr_f] #filt.shape = 10,10
                    image_test = image[curr_y:curr_y+f, curr_x:curr_x+f, 0] # this.shape = (10,10)
                    a = matrix_multiply(filt, image[curr_y:curr_y+f, curr_x:curr_x+f, 0])
      
                    out = 0

                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
        
        return a

        # return tf.matmul(inputs, self.w) 

