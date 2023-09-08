from utility import *
from replay_buffer import ReplayBuffer
from objective_and_gradient import OB_and_GD
from rollout import rollout_one_step_with_noise
from environment.TIPenv import TIPSystem
from actor_network import GAUSSIAN_POLICY
from critic_network import Q_NETWORK
from evaluation import EVAL
import pandas as pd
import os

flags = tf.flags
FLAGS = flags.FLAGS

''' ENVIRONMENT '''
# Set environment here
flags.DEFINE_string("env_name", "TIP_SYM", "Environment name")
flags.DEFINE_boolean("cuda", False, "If True, use GPU device")
flags.DEFINE_string("phase", "test", "Environment name")

''' ALGORITHM '''
flags.DEFINE_string("random_seed", "1", "Random seed")
flags.DEFINE_string("name", "ver", "File name for saving the results of evaluation and parameters")

''' MONITORING AND RENDERING '''
flags.DEFINE_boolean("monitoring", True, "If True, the results of evaluation are saved as a csv file")
flags.DEFINE_boolean("rendering", True, "If True, rendering is executed at the last of every evaluation stages")

''' META PARAMETERS '''
flags.DEFINE_integer("max_interaction", 2000000, "Maximum value of training iterations")
flags.DEFINE_integer("max_step", 1000, "Cutoff for continuous task. Default value is the value defined in the environment")
flags.DEFINE_integer("random_step", 1000, "The number of initial exploration steps")
flags.DEFINE_integer("eval_period", 5000, "Policy is frequently evaluated after experiencing this number of episodes")
flags.DEFINE_integer("num_eval", 10, "Policy is evaluated by testing this number of episodes and averaging the total reward")
flags.DEFINE_float("gamma", 0.99, "Discounting factor")
flags.DEFINE_float("reward_scale", 1.0, "Scaling reward by this factor")

''' DATA HANDLING '''
flags.DEFINE_integer("buffer_size", 1000000, "Buffer size")
flags.DEFINE_integer("batch_size", 256, "Batch size for each updates")
flags.DEFINE_integer("replay_start_size", 1000, "Buffer size for starting updates")

''' UPDATE '''
flags.DEFINE_float("target_val_alpha", 0.995, "Exponential moving decay rate for target critic update")
flags.DEFINE_integer("num_update", 1, "The number of updating networks per iteration")
flags.DEFINE_float("pol_lr", 3e-4, "Learning rate for policy network")
flags.DEFINE_float("val_lr", 3e-4, "Learning rate for Q network")
flags.DEFINE_float("alpha_lr", 3e-4, "Learning rate for Q network")

seed = int(FLAGS.random_seed) + 1
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
if FLAGS.cuda == False:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.reset_default_graph()
np.random.seed(seed)
tf.random.set_random_seed(seed)

env = TIPSystem()
eval_env = TIPSystem()

class SAC_TRAINER:

    def __init__(self, env, eval_env, FLAGS):
        self.env = env
        self.eval_env = eval_env

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=tf.get_default_graph(),config=config)
        self.state_input=tf.placeholder(tf.float32,shape=[None,self.env.state_dim])
        self.next_state_input = tf.placeholder(tf.float32, shape=[None, self.env.state_dim])
        self.action_input = tf.placeholder(tf.float32, shape=[None, self.env.action_dim])

        self.root_model = "./checkpoint/%s/%s/" % (FLAGS.env_name, FLAGS.name)
        self.root_learning_curve = "./learning_curve/%s/%s/" % (FLAGS.env_name, FLAGS.name)
        self.root_param = self.root_model + "parameter/"
        self.root_buffer = self.root_model + "buffer/"
        if not os.path.isdir("./checkpoint/%s"%FLAGS.env_name) : os.mkdir("./checkpoint/%s"%FLAGS.env_name)
        if not os.path.isdir("./learning_curve/%s" % FLAGS.env_name): os.mkdir("./learning_curve/%s" % FLAGS.env_name)
        if not os.path.isdir(self.root_model[:-1]): os.mkdir(self.root_model[:-1])
        if not os.path.isdir(self.root_learning_curve[:-1]): os.mkdir(self.root_learning_curve[:-1])
        if not os.path.isdir(self.root_param[:-1]): os.mkdir(self.root_param[:-1])
        if not os.path.isdir(self.root_buffer[:-1]): os.mkdir(self.root_buffer[:-1])

        if FLAGS.phase == "train":
            self.monitoring_file = self.root_learning_curve+"learning_curve.csv"
            file = pd.DataFrame(columns=["Episode", "Step", "Max", "Min", "Average"])
            file.to_csv(self.monitoring_file, index=False)

        self.FLAGS = FLAGS
        self.state = self.env.reset()
        self.replay_buffer = ReplayBuffer(self.env.state_dim, self.env.action_dim, FLAGS.buffer_size, self.root_buffer)
        self.global_step = 0
        self.local_step = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.define_actor_and_critic_network()
        self.define_loss_and_gradients()
        self.define_training_method()
        self.define_loop_training_method()

        self.mean_s = np.zeros(self.env.state_dim)
        self.mean_s2 = np.zeros(self.env.state_dim)
        self.count = 0

    ''' DEFINE Networks and Objective '''

    def define_actor_and_critic_network(self):
        self.actor = GAUSSIAN_POLICY(sess=self.sess,
                                     state_input=self.state_input,
                                     action_dim=self.env.action_dim,
                                     name="actor",
                                     reuse=False)

        self.target_actor = GAUSSIAN_POLICY(sess=self.sess,
                                            state_input=self.next_state_input,
                                            action_dim=self.env.action_dim,
                                            name="actor",
                                            reuse=True)

        self.critic1 = Q_NETWORK(state_input=self.state_input,
                                 action_input=self.action_input,
                                 actor_input=self.actor.action,
                                 dummy_actor_input=self.actor.dummy_action,
                                 random_seed=seed,
                                 name="qf1")

        self.critic2 = Q_NETWORK(state_input=self.state_input,
                                 action_input=self.action_input,
                                 actor_input=self.actor.action,
                                 dummy_actor_input=self.actor.dummy_action,
                                 random_seed=seed,
                                 name="qf2")

        self.target_critic1 = Q_NETWORK(state_input=self.next_state_input,
                                        action_input=self.action_input,
                                        actor_input=self.target_actor.action,
                                        dummy_actor_input=self.target_actor.dummy_action,
                                        random_seed=seed,
                                        name="target_qf1")

        self.target_critic2 = Q_NETWORK(state_input=self.next_state_input,
                                        action_input=self.action_input,
                                        actor_input=self.target_actor.action,
                                        dummy_actor_input=self.target_actor.dummy_action,
                                        random_seed=seed,
                                        name="target_qf2")

    def define_loss_and_gradients(self):
        self.obg = OB_and_GD(actor=self.actor,
                             target_actor=self.target_actor,
                             critic=self.critic1,
                             target_critic=self.target_critic1,
                             critic2=self.critic2,
                             target_critic2=self.target_critic2,
                             user_set=self.FLAGS)
        self.critic_loss = self.obg.critic_loss()
        self.pi_loss, self.actor_grad = self.obg.actor_gradients()
        self.alpha_loss = self.obg.alpha_loss()

    def define_training_method(self):
        self.actor_trainer = tf.train.AdamOptimizer(self.FLAGS.pol_lr)
        self.actor_train = self.actor_trainer.apply_gradients(zip(self.actor_grad,self.actor.params))
        self.critic_trainer = tf.train.AdamOptimizer(self.FLAGS.val_lr)
        self.critic_train = self.critic_trainer.minimize(self.critic_loss)
        self.alpha_trainer = tf.train.AdamOptimizer(self.FLAGS.alpha_lr)
        self.alpha_train = self.alpha_trainer.minimize(self.alpha_loss)
        self.target_critic_init = [i.assign(j) for i, j in zip(self.target_critic1.params, self.critic1.params)]+ \
                                  [i.assign(j) for i, j in zip(self.target_critic2.params, self.critic2.params)]

        target_update = []
        for v_source, v_target in zip(self.critic1.params+self.critic2.params, self.target_critic1.params+self.target_critic2.params):
            update_op = v_target.assign_sub((1 - self.FLAGS.target_val_alpha) * (v_target - v_source))
            target_update.append(update_op)
        self.target_critic_train = tf.group(*target_update)

    def define_loop_training_method(self):
        with tf.control_dependencies([self.critic_train]):
            actor_train = self.actor_trainer.apply_gradients(zip(self.actor_grad,self.actor.params))
        with tf.control_dependencies([actor_train]):
            alpha_train = self.alpha_trainer.minimize(self.alpha_loss)
        with tf.control_dependencies([alpha_train]):
            target_update = []
            for v_source, v_target in zip(self.critic1.params + self.critic2.params,
                                          self.target_critic1.params + self.target_critic2.params):
                update_op = v_target.assign_sub((1 - self.FLAGS.target_val_alpha) * (v_target - v_source))
                target_update.append(update_op)
            target_critic_train = tf.group(*target_update)
        with tf.control_dependencies([target_critic_train]):
            self.loop_train = tf.identity(self.pi_loss)

    def save_net(self):
        self.saver.save(self.sess, self.root_model+self.FLAGS.name+".ckpt")
        print("Networks %s - %s have been successfully saved"%(self.FLAGS.env_name,self.FLAGS.name))

    def load_net(self, params):
        try:
            saver2 = tf.train.Saver(params)
            saver2.restore(self.sess, self.root_model+self.FLAGS.name+".ckpt")
            print("Networks have been successfully restored")
        except:
            raise ValueError("Can not find old networks")

    ''' OPERATIONS '''

    def rollout(self, random_action):
        self.state, data, self.local_step, self.episode_step, self.episode_reward, self.terminal \
            = rollout_one_step_with_noise(env=self.env,
                                          state=self.state,
                                          actor=self.actor,
                                          max_step=self.FLAGS.max_step,
                                          rwd_scale=self.FLAGS.reward_scale,
                                          local_step=self.local_step,
                                          episode_step=self.episode_step,
                                          episode_reward=self.episode_reward,
                                          random_action=random_action)

        self.replay_buffer.add(data["obs"], data["act"], data["rew"], data["next_obs"], data["done"])

        if self.terminal:
            self.episode_reward = 0.0

    def train(self):

        for i in range(1):
            minibatch = self.replay_buffer.sample(self.FLAGS.batch_size)
            state_batch = minibatch["obs"]
            action_batch = minibatch["action"]
            reward_batch = minibatch["reward"]
            next_state_batch = minibatch["next_obs"]
            done_batch = minibatch["done"]
            feed_dict = {
                self.state_input: state_batch,
                self.action_input: action_batch,
                self.next_state_input: next_state_batch,
                self.obg.reward_input: reward_batch,
                self.obg.done_input: done_batch
            }
            self.sess.run(self.critic_train,feed_dict=feed_dict)

        minibatch = self.replay_buffer.sample(self.FLAGS.batch_size)
        state_batch = minibatch["obs"]
        action_batch = minibatch["action"]
        reward_batch = minibatch["reward"]
        next_state_batch = minibatch["next_obs"]
        done_batch = minibatch["done"]
        feed_dict = {
            self.state_input: state_batch,
            self.action_input: action_batch,
            self.next_state_input: next_state_batch,
            self.obg.reward_input: reward_batch,
            self.obg.done_input: done_batch
        }
        self.sess.run(self.loop_train, feed_dict=feed_dict)

    def evaluation(self):
        self.avg_eval = EVAL(env=self.eval_env,
                             actor=self.actor,
                             cur_epi=self.episode_step,
                             max_step=self.FLAGS.max_step,
                             num_eval=self.FLAGS.num_eval,
                             rendering=self.FLAGS.rendering,
                             monitoring=self.FLAGS.monitoring,
                             file_path=self.monitoring_file,
                             global_step=self.global_step)

    ''' MAIN Code '''

    def execution(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_critic_init)
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        while self.global_step < self.FLAGS.max_interaction:
            if self.global_step < self.FLAGS.random_step:
                self.rollout(random_action=True)
            else:
                self.rollout(random_action=False)
            self.global_step += 1
            if self.global_step >= self.FLAGS.replay_start_size:
                for i in range(self.FLAGS.num_update):
                    self.train()
            if self.global_step%self.FLAGS.eval_period==0:
                self.evaluation()
                self.save_net()
                self.replay_buffer.save()

        # Save weights
        self.save_net()
        self.replay_buffer.save()

        # Finish notification
        finish_notice(self.root_param)
        self.sess.close()

    def testing(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_critic_init)
        self.load_net(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        for epoch in range(10):
            state = self.eval_env.reset()
            ep_reward = 0
            for j in range(self.FLAGS.max_step):
                self.eval_env.render()
                action = np.clip(self.actor.ACTION_EVAL(state)[0], -1.0, 1.0)
                next_state, reward, done, info = self.eval_env.step(self.eval_env.action_max*action)
                state = next_state
                ep_reward += reward
                if done:
                    break
            print("[EPI#%d] | TR %.2f"%(epoch+1,ep_reward))

if __name__ == "__main__":
    TRAINING_AGENT = SAC_TRAINER(env, eval_env, FLAGS)
    if FLAGS.phase == "train":
        TRAINING_AGENT.execution()
    else:
        TRAINING_AGENT.testing()