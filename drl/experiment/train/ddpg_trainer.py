import torch
import numpy as np
from collections import deque

from drl.env.i_environment import IEnvironment
from drl.experiment.configuration import Configuration
from drl.experiment.train.trainer import Trainer


class DdpgTrainer(Trainer):

    def __init__(self, cfg: Configuration, session_id, path_models='models'):

        super(DdpgTrainer, self).__init__(cfg, session_id)

    def train(self, agent, env: IEnvironment, model_filename=None):

        trainer_cfg = self.cfg.get_current_exp_cfg().trainer_cfg

        # n_episodes = 2000
        # max_t = 700

        n_episodes = trainer_cfg.max_steps
        max_t = trainer_cfg.max_episode_steps

        scores_deque = deque(maxlen=100)
        scores = []
        max_score = -np.Inf
        for i_episode in range(1, n_episodes+1):
            # state = env.reset()
            state, new_life = env.reset()
            agent.reset()
            score = 0
            for t in range(max_t):
                action = agent.act(state)
                # next_state, reward, done, _ = env.step(action)
                next_state, reward, done, new_life = env.step(action)

                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_deque.append(score)
            scores.append(score)
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
            if i_episode % 100 == 0:
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        return scores
