import numpy as np

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), path=None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.path = path

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        minibatch = {}
        minibatch["obs"] = self.state[ind].reshape([batch_size,-1])
        minibatch["action"] = self.action[ind].reshape([batch_size,-1])
        minibatch["reward"] = self.reward[ind].reshape([batch_size,-1])
        minibatch["next_obs"] = self.next_state[ind].reshape([batch_size,-1])
        minibatch["done"] = self.done[ind].reshape([batch_size,-1])
        return minibatch

    def save(self):
        assert self.path != None
        np.save(self.path + "state.npy", self.state[:self.size])
        np.save(self.path + "next_state.npy", self.next_state[:self.size])
        np.save(self.path + "action.npy", self.action[:self.size])
        np.save(self.path + "reward.npy", self.reward[:self.size])
        np.save(self.path + "done.npy", self.done[:self.size])
        np.save(self.path + "ptr.npy", np.array([self.ptr]))

    def load(self):
        assert self.path != None
        state = np.load(self.path + "state.npy")
        next_state = np.load(self.path + "next_state.npy")
        action = np.load(self.path + "action.npy")
        reward = np.load(self.path + "reward.npy")
        done = np.load(self.path + "done.npy")
        ptr = np.load(self.path + "ptr.npy")
        self.size = len(state)
        self.state[:self.size] = state
        self.next_state[:self.size] = next_state
        self.action[:self.size] = action
        self.reward[:self.size] = reward.reshape([-1,1])
        self.done[:self.size] = done.reshape([-1,1])
        self.ptr = ptr[0]