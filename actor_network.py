from utility import *
import tensorflow_probability as tfp

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class GAUSSIAN_POLICY(object):

    def __init__(self, sess, state_input, action_dim, name, random_seed=1, reuse=False):
        self.sess = sess
        self.state_input = state_input
        self.action_dim = action_dim

        Layer = Fully_connected(name + "/FC1", state_input, out_dim=400, activation=tf.nn.relu,
                                kernel_init=tf.contrib.layers.xavier_initializer(seed=random_seed), reuse=reuse)
        Layer = Fully_connected(name + "/FC2", Layer, out_dim=300, activation=tf.nn.relu,
                                kernel_init=tf.contrib.layers.xavier_initializer(seed=random_seed), reuse=reuse)
        Layer = Fully_connected(name + "/FC3", Layer, out_dim=action_dim*2,
                                kernel_init=tf.contrib.layers.xavier_initializer(seed=random_seed), reuse=reuse)
        self.mu = Layer[..., :self.action_dim]
        self.log_sigma = tf.clip_by_value(Layer[..., self.action_dim:], LOG_SIG_MIN, LOG_SIG_MAX)
        self.sigma = tf.exp(self.log_sigma)
        self.dist = tfp.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=self.sigma)

        self.sample = self.dist.sample()
        self.action = tf.tanh(self.sample)
        self.eval_action = tf.tanh(self.mu)
        self.log_pi = tf.reshape(
            self.dist.log_prob(self.sample) - tf.reduce_sum(tf.log(1 - (self.action ** 2) + 1e-10), axis=1), [-1, 1])

        ###################
        dummy_state = tf.constant([-1.,-1.,-1.,-1.,1.,1.,1.,-1.,-1.,-1.,-1.])
        Layer = Fully_connected(name + "/FC1", dummy_state*state_input, out_dim=400, activation=tf.nn.relu,
                                kernel_init=tf.contrib.layers.xavier_initializer(seed=random_seed), reuse=True)
        Layer = Fully_connected(name + "/FC2", Layer, out_dim=300, activation=tf.nn.relu,
                                kernel_init=tf.contrib.layers.xavier_initializer(seed=random_seed), reuse=True)
        Layer = Fully_connected(name + "/FC3", Layer, out_dim=action_dim*2,
                                kernel_init=tf.contrib.layers.xavier_initializer(seed=random_seed), reuse=True)
        mu = Layer[..., :self.action_dim]
        log_sigma = tf.clip_by_value(Layer[..., self.action_dim:], LOG_SIG_MIN, LOG_SIG_MAX)
        sigma = tf.exp(log_sigma)
        dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

        sample = dist.sample()
        self.dummy_action = tf.tanh(sample)
        self.dummy_log_pi = tf.reshape(
            dist.log_prob(sample) - tf.reduce_sum(tf.log(1 - (self.dummy_action ** 2) + 1e-10), axis=1), [-1, 1])
        ################

        params = Get_param(scope=name, type='trainable')
        self.weight = []
        self.bias = []
        for par in params:
            if 'kernel' in par.name:
                self.weight.append(par)
            else:
                self.bias.append(par)
        self.params = self.weight + self.bias

    def ACTION_SAMPLE(self, state):
        eps = np.random.uniform(0.0,1.0)
        if eps < 0.5:
            return self.sess.run(self.action, feed_dict={self.state_input: state})
        else:
            return -self.sess.run(self.dummy_action, feed_dict={self.state_input: state})

    def ACTION_EVAL(self, state):
        return self.sess.run(self.eval_action, feed_dict={self.state_input: state})