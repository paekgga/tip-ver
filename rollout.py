from utility import *
from copy import deepcopy

def rollout_one_step_with_noise(env, state, actor, max_step, episode_reward,
                                local_step, episode_step, random_action=False, rwd_scale=1.0):
    episode_step_ = deepcopy(episode_step)
    local_step_ = deepcopy(local_step)
    episode_reward_ = deepcopy(episode_reward)
    data = {}
    if random_action:
        action = np.random.uniform(-1.,1.,env.action_dim)
    else:
        action = np.clip(actor.ACTION_SAMPLE(state)[0],-1.0,1.0)
    next_state, reward, terminal, _ = env.step(env.action_max*action)
    episode_reward_ += reward
    reward *= rwd_scale
    local_step_ += 1
    if local_step_>=max_step: done = False
    else: done = terminal
    data["obs"] = state.flatten()
    data["next_obs"] = next_state.flatten()
    data["act"] = action
    data["rew"] = reward
    data["done"] = done
    if terminal or local_step_>=max_step:
        terminal = True
        next_state = env.reset()
        local_step_ = 0
        episode_step_ += 1
    return next_state, data, local_step_, episode_step_, episode_reward_, terminal