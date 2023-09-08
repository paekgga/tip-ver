import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np

def Convolution(name, input, n_filter, k_size, strides, padding="SAME",activation=tf.identity, use_bias=True,
                reuse=False, trainable=True, kernel_init=tf.contrib.layers.xavier_initializer(),
                bias_init=tf.zeros_initializer()):
    with tf.variable_scope(name) as scope:
        if reuse: scope.reuse_variables()
        out = tf.layers.conv2d(input, n_filter, k_size, strides=strides, padding=padding,
                               activation=activation, use_bias=use_bias, trainable=trainable,
                               kernel_initializer=kernel_init, bias_initializer=bias_init)
    kernel = [tensor for tensor in Get_param(scope=name, type="global") if tensor.name.endswith("kernel:0")][0]
    tf.add_to_collection("L2_kernel", tf.reduce_sum(tf.square(kernel)))
    if use_bias:
        bias = [tensor for tensor in Get_param(scope=name, type="global") if tensor.name.endswith("bias:0")][0]
        tf.add_to_collection("L2_bias", tf.reduce_sum(tf.square(bias)))
    return out

def Deconvolution(name, input, n_filter, k_size, strides, padding="SAME",activation=tf.identity, use_bias=True,
                  reuse=False, trainable=True, kernel_init=tf.contrib.layers.xavier_initializer(),
                bias_init=tf.zeros_initializer()):
    with tf.variable_scope(name) as scope:
        if reuse: scope.reuse_variables()
        out = tf.layers.conv2d_transpose(input,n_filter,k_size,strides=strides,padding=padding,
                                         activation=activation,use_bias=use_bias,trainable=trainable,
                                         kernel_initializer=kernel_init, bias_initializer=bias_init)
    kernel = [tensor for tensor in Get_param(scope=name,type="global") if tensor.name.endswith("kernel:0")][0]
    tf.add_to_collection("L2_kernel", tf.reduce_sum(tf.square(kernel)))
    if use_bias:
        bias = [tensor for tensor in Get_param(scope=name,type="global") if tensor.name.endswith("bias:0")][0]
        tf.add_to_collection("L2_bias", tf.reduce_sum(tf.square(bias)))
    return out

def LRN(input, depth_radius):
    return tf.nn.lrn(input, depth_radius=depth_radius)

def Pooling(input, k_size, strides, padding='SAME'):
    return tf.nn.max_pool(input, ksize=k_size, strides=strides, padding=padding)

def Fully_connected(name, input, out_dim, activation=tf.identity, reuse=False, use_bias=True, bayesian_prob=None,
                    trainable=True, kernel_init=tf.contrib.layers.xavier_initializer(), bias_init=tf.zeros_initializer()):
    with tf.variable_scope(name) as scope:
        if reuse: scope.reuse_variables()
        if bayesian_prob == None:
            out = tf.layers.dense(input, out_dim, activation=activation, use_bias=use_bias,trainable=trainable,
                                  kernel_initializer=kernel_init, bias_initializer=bias_init)
            kernel = [tensor for tensor in Get_param(scope=name,type="global") if tensor.name.endswith("kernel:0")][0]
            if use_bias:
                bias = [tensor for tensor in Get_param(scope=name,type="global") if tensor.name.endswith("bias:0")][0]
        else:
            kernel = tf.get_variable("dense/kernel",shape=[input.get_shape()[1], out_dim],
                                     initializer=kernel_init,trainable=trainable)
            if use_bias:
                bias = tf.get_variable("dense/bias",shape=[out_dim], initializer=bias_init,trainable=trainable)
            else:
                bias = 0
            bern = tf.distributions.Bernoulli(probs=bayesian_prob,dtype=tf.float32)
            weight = tf.matmul(tf.diag(bern.sample([input.get_shape()[1],])), kernel)
            out = activation(tf.matmul(input, weight) + bias)
            tf.add_to_collection("BAYES_REG",bayesian_prob*tf.reduce_sum(tf.square(kernel))+tf.reduce_sum(tf.square(bias)))
    tf.add_to_collection("L2_kernel", tf.reduce_sum(tf.square(kernel)))
    if use_bias:
        tf.add_to_collection("L2_bias", tf.reduce_sum(tf.square(bias)))
    return out

def Flatten(input):
    return tf.contrib.layers.flatten(input)

def Concat(source_list,axis=1):
    return tf.concat(source_list,axis=axis)

def Dropout(input, keep_prob):
    return tf.nn.dropout(input, keep_prob)

def BatchNorm(name, input, is_training, activation=tf.identity, reuse=False, trainable=True):
    with tf.variable_scope(name) as scope:
        if reuse: scope.reuse_variables()
        out = tf.layers.batch_normalization(input,training=is_training,reuse=reuse,trainable=trainable)
    return activation(out)

def NP_onehot(a, num_class, on_value=1.0, off_value=0.0):
    onehot = np.squeeze(np.eye(num_class)[a.reshape([-1]).astype(np.int32)])
    return np.where(onehot==1.0,on_value,off_value)

def TF_onehot(a, num_class, on_value=1.0, off_value=0.0):
    return tf.one_hot(a, num_class, on_value=on_value, off_value=off_value)

def Get_param(scope=None, type="trainable"):
    if type=="trainable":
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
    else:
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)

def Get_kernel(scope=None, type="trainable"):
    if type=="trainable":
        set = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
    else:
        set = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)
    return [tensor for tensor in set if tensor.name.endswith("kernel:0")]

def Get_bias(scope=None, type="trainable"):
    if type=="trainable":
        set = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
    else:
        set = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)
    return [tensor for tensor in set if tensor.name.endswith("bias:0")]

def Get_batch(idx, itr, X_data, Y_data=None, batch_size=128):
    if itr == int(len(idx) / batch_size) - 1:
        if Y_data is not None:
            return X_data[idx[itr * batch_size:]], Y_data[idx[itr * batch_size:]]
        else:
            return X_data[idx[itr * batch_size:]]
    else:
        if Y_data is not None:
            return X_data[idx[itr * batch_size:(itr + 1) * batch_size]], Y_data[idx[itr * batch_size:(itr + 1) * batch_size]]
        else:
            return X_data[idx[itr * batch_size:(itr + 1) * batch_size]]

def Save_network(sess, model_name):
    saver = tf.train.Saver()
    path = "TRAINED_MODEL/%s.ckpt"%(model_name)
    saver.save(sess,path)
    print("\nNetwork is successfully saved as [%s]\n"%path)

def Load_network(sess, model_name):
    saver = tf.train.Saver()
    path = "TRAINED_MODEL/%s.ckpt"%(model_name)
    saver.restore(sess,path)
    print("\nNetwork [%s] is successfully restored\n"%path)

def L2_kernel_loss():
    return tf.add_n(tf.get_collection("L2_kernel"))

def L2_bias_loss():
    return tf.add_n(tf.get_collection("L2_bias"))

def L2_loss():
    return tf.add_n(tf.get_collection("L2_kernel"))+tf.add_n(tf.get_collection("L2_bias"))

def Bayesian_regularizer():
    return tf.add_n(tf.get_collection("BAYES_REG"))

class VariationalDenseGaussian:
    def __init__(self, n_inp, n_out, name, scope_sub, use_bias=True, criteria=3.0, reuse=False,
                 kernel_init=tf.contrib.layers.xavier_initializer()):
        self.use_bias = use_bias
        self.n_out = n_out
        with tf.variable_scope(name) as scope:
            if reuse: scope.reuse_variables()
            self.theta_ = tf.get_variable("theta/" + scope_sub, [n_inp, n_out],
                                          initializer=kernel_init)
            self.theta = tf.where(tf.is_nan(self.theta_), tf.zeros_like(self.theta_), self.theta_)
            self.log_sigma2_ = tf.get_variable("log_sigma2/"+scope_sub, [n_inp, n_out],
                                              initializer=tf.constant_initializer(-10.))
            self.log_sigma2 = tf.where(tf.is_nan(self.log_sigma2_), tf.zeros_like(self.log_sigma2_), self.log_sigma2_)
            self.sigma2 = tf.exp(self.log_sigma2)
            if self.use_bias:
                self.bias_ = tf.get_variable("bias/" + scope_sub, [n_out],
                                             initializer=tf.zeros_initializer())
                self.bias = tf.where(tf.is_nan(self.bias_), tf.zeros_like(self.bias_), self.bias_)
            self.log_alpha = tf.clip_by_value(self.log_sigma2 - tf.log(tf.square(self.theta)), -20.0, 4.0)
            self.log_alpha = tf.where(tf.is_nan(self.log_alpha), 4. * tf.ones_like(self.log_alpha), self.log_alpha)
            self.boolean_mask = self.log_alpha<=criteria
            self.sigma = tf.sqrt(tf.exp(self.log_alpha)*self.theta*self.theta)
            self.weight_mask = tf.cast(self.boolean_mask, tf.int32)

    def mask_processing(self, next_mask):
        if next_mask is not None:
            consider_next = tf.cast(tf.reduce_sum(next_mask,axis=1)>0,tf.int32)
            self.weight_mask = self.weight_mask*consider_next
        self.count = tf.count_nonzero(self.weight_mask)
        self.sparse_theta = tf.where(self.weight_mask>0, self.theta, tf.zeros_like(self.theta))

    def __call__(self, input, input_mean, input_sparse, activation=tf.identity):
        sample_mean = tf.matmul(input, self.theta)
        sample_std = tf.sqrt(tf.matmul(tf.square(input),self.sigma2))
        sample_ = sample_mean + sample_std*tf.random_normal(tf.shape(sample_mean),0.0,1.0,seed=500)

        avg_ = tf.matmul(input_mean, self.theta)
        sparse_ = tf.matmul(input_sparse, self.sparse_theta)

        if self.use_bias:
            output = activation(sample_+self.bias)
            output_mean = activation(avg_+self.bias)
            output_sparse = activation(sparse_+self.bias)
        else:
            output = activation(sample_)
            output_mean = activation(avg_)
            output_sparse = activation(sparse_)
        return output, output_mean, output_sparse

    @property
    def regularization(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        mdkl = k1*tf.sigmoid(k2 + k3*self.log_alpha)-0.5*tf.log(1+(tf.exp(-self.log_alpha))) + C
        return -tf.reduce_sum(mdkl)

def intro_notice(FLAGS, root, root2):
    print("\n#################################################################################")
    print("\n[Env_name]: ",FLAGS.env_name)
    print("[Algorithm]: ", FLAGS.algorithm)
    print("[File_name]: ", FLAGS.file_name)
    print("\n[Notice]")
    if FLAGS.monitoring == True:
        print(" - Evaluation history will be saved as: "+root+"%s.csv"%FLAGS.file_name)
    if FLAGS.algorithm == "DDPG":
        print(" - Trained parameters will be saved as: " + root2 + "WX_%s.npy" % FLAGS.file_name)
    else:
        print(" - Trained sparse parameters will be saved as: " + root2 + "WX_sparse_%s.npy" % FLAGS.file_name)
        print(" - Compressed parameters will be saved as: " + root2 + "WX_compressed_%s.npy" % FLAGS.file_name)
    if FLAGS.monitoring == True:
        print(" - You can monitor learning curve by running 'monitoring.py' or 'monitoring_with_sparsity.py'")
        print("   (It requires typing [File_name] in 'monitoring.py'")
        print(" - After training, you can validate the performance of the trained policy by running 'validation.py'")
        print("   (It requires typing path to the saved parameters in 'validation.py'")
        print("\n#################################################################################\n")

def finish_notice(root):
    print("\n#################################################################################")
    print("\nTraining is finished")
    print(" - Parameters are successfully saved at " + root)
    print(" - You can run 'validation.py' with the saved parameters "
          "for testing the performance of the trained policy")
    print("\n#################################################################################\n")