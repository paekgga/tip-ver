from utility import *

class Q_NETWORK(object):

    def __init__(self, state_input, action_input, actor_input=None, dummy_actor_input=None, random_seed=1, name="Qnet"):

        Layer = Concat([state_input,action_input])
        Layer = Fully_connected(name + "/FC1", Layer, out_dim=400, activation=tf.nn.relu,
                                kernel_init=tf.contrib.layers.xavier_initializer(seed=random_seed))
        Layer = Fully_connected(name + "/FC2", Layer, out_dim=300, activation=tf.nn.relu,
                                kernel_init=tf.contrib.layers.xavier_initializer(seed=random_seed))
        self.q = Fully_connected(name + "/FC3", Layer, out_dim=1,
                                kernel_init=tf.contrib.layers.xavier_initializer(seed=random_seed))

        dummy_state = tf.constant([-1.,-1.,-1.,-1.,1.,1.,1.,-1.,-1.,-1.,-1.])
        Layer = Concat([dummy_state*state_input, -action_input])
        Layer = Fully_connected(name + "/FC1", Layer, out_dim=400, activation=tf.nn.relu, reuse=True)
        Layer = Fully_connected(name + "/FC2", Layer, out_dim=300, activation=tf.nn.relu, reuse=True)
        self.q_dummy = Fully_connected(name + "/FC3", Layer, out_dim=1, reuse=True)

        if actor_input != None:
            Layer = Concat([state_input, actor_input])
            Layer = Fully_connected(name + "/FC1", Layer, out_dim=400, activation=tf.nn.relu, reuse=True)
            Layer = Fully_connected(name + "/FC2", Layer, out_dim=300, activation=tf.nn.relu, reuse=True)
            self.q_actor = Fully_connected(name + "/FC3", Layer, out_dim=1, reuse=True)
            self.params = Get_param(scope=name, type='trainable')

            Layer = Concat([dummy_state*state_input, dummy_actor_input])
            Layer = Fully_connected(name + "/FC1", Layer, out_dim=400, activation=tf.nn.relu, reuse=True)
            Layer = Fully_connected(name + "/FC2", Layer, out_dim=300, activation=tf.nn.relu, reuse=True)
            self.q_dummy_actor = Fully_connected(name + "/FC3", Layer, out_dim=1, reuse=True)