from utility import *
import pandas as pd

def EVAL(env,actor,cur_epi,max_step,num_eval=10,rendering=False,
         monitoring=False,file_path=None,global_step=0):

    episode_rewards = []
    total_step = 0
    for ii in range(num_eval):
        state = env.reset()
        ep_reward = 0
        for j in range(max_step):
            if ii==num_eval-1 and cur_epi!=0 and rendering == True:
                env.render()
            action = np.clip(actor.ACTION_EVAL(state)[0], -1.0, 1.0)
            next_state, reward, done, info = env.step(env.action_max*action)
            state = next_state
            ep_reward += reward
            total_step += 1
            if done:
                break
        episode_rewards.append(ep_reward)

    max_eval = np.max(episode_rewards)
    min_eval = np.min(episode_rewards)
    avg_eval = np.mean(episode_rewards)

    if monitoring == True and file_path != None:
        file = pd.read_csv(file_path)
        file.loc[len(file)] = [cur_epi, global_step, max_eval, min_eval, avg_eval]
        file.to_csv(file_path, index=False)

    print("[#Epi]: ", cur_epi, " | [#Step]: ",global_step,
          " | [Max]: %.3f"%max_eval, " | [Min]: %.3f"%min_eval,
          " | [Average]: %.3f"%avg_eval)

    return avg_eval