from utility import *
import tensorflow_probability as tfp

class OB_and_GD(object):

    def __init__(self, actor, target_actor, critic, critic2, target_critic, target_critic2, user_set):
        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic
        self.critic2 = critic2
        self.target_critic2 = target_critic2
        self.gamma = user_set.gamma
        self.user_set = user_set

        self.log_alpha = tf.Variable(0.0)
        self.alpha = tf.exp(self.log_alpha)

        self.reward_input = tf.placeholder(tf.float32, shape=[None, 1])
        self.done_input = tf.placeholder(tf.float32, shape=[None, 1])

    def critic_loss(self):
        q_actor = tf.minimum(self.target_critic.q_actor, self.target_critic2.q_actor)
        minq = q_actor - self.alpha*self.target_actor.log_pi
        targets = self.reward_input + self.gamma * (1 - self.done_input) * minq

        delta1 = tf.stop_gradient(targets) - self.critic.q
        delta2 = tf.stop_gradient(targets) - self.critic2.q
        critic_loss1 = 0.5 * tf.reduce_mean((tf.square(delta1) + tf.square(delta2)))

        # q_actor = tf.minimum(self.target_critic.q_dummy_actor, self.target_critic2.q_dummy_actor)
        # minq = q_actor - self.alpha*self.target_actor.dummy_log_pi
        # targets = self.reward_input + self.gamma * (1 - self.done_input) * minq

        delta1 = tf.stop_gradient(targets) - self.critic.q_dummy
        delta2 = tf.stop_gradient(targets) - self.critic2.q_dummy
        critic_loss2 = 0.5 * tf.reduce_mean((tf.square(delta1) + tf.square(delta2)))

        critic_loss = (critic_loss1 + critic_loss2)/2

        return critic_loss

    def actor_gradients(self):
        q1 = self.critic.q_actor
        q2 = self.critic2.q_actor
        min_q = tf.minimum(q1, q2)
        pi_loss1 = tf.reduce_mean((self.alpha*self.actor.log_pi - min_q), axis=0)

        q1 = self.critic.q_dummy_actor
        q2 = self.critic2.q_dummy_actor
        min_q = tf.minimum(q1, q2)
        pi_loss2 = tf.reduce_mean((self.alpha*self.actor.dummy_log_pi - min_q), axis=0)

        pi_loss = (pi_loss1 + pi_loss2)/2

        actor_grad = tf.gradients(pi_loss, self.actor.params)

        return pi_loss, actor_grad

    def alpha_loss(self):
        alpha_loss = tf.reduce_mean(-self.alpha * (tf.stop_gradient(self.actor.log_pi) - self.actor.action_dim))
        return alpha_loss