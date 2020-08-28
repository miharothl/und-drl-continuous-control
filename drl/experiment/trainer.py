import os
from pathlib import Path

import torch
import numpy as np
from collections import deque
import sys
import logging

from drl.env.i_environment import IEnvironment
from drl.experiment.configuration import Configuration
from drl.experiment.config.trainer_config import TrainerConfig
from drl.experiment.recorder import Recorder


class Trainer:
    def __init__(self, config: Configuration, session_id, path_models='models'):
        self.__config = config
        self.__session_id = session_id

    def get_model_filename(self, episode, score, val_score, eps):

        session_path = os.path.join(self.__config.get_app_experiments_path(train_mode=True), self.__session_id)
        Path(session_path).mkdir(parents=True, exist_ok=True)

        import re
        model_id = re.sub('[^0-9a-zA-Z]+', '', self.__config.get_current_exp_cfg().id)
        model_id = model_id.lower()
        filename = "{}_{}_{}_{:.2f}_{:.2f}_{:.2f}.pth".format(model_id, self.__session_id, episode, score, val_score,
                                                              eps)

        model_path = os.path.join(session_path, filename)

        return model_path

    def select_model_filename(self, model_filename=None):
        if model_filename is not None:
            path = os.path.join(self.__path_models, model_filename)
            return path

    def train(self, agent, env: IEnvironment, model_filename=None):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """

        trainer_cfg: TrainerConfig = self.__config.get_current_exp_cfg().trainer_cfg

        max_steps = trainer_cfg.max_steps
        max_episode_steps = trainer_cfg.max_episode_steps

        eval_frequency = trainer_cfg.eval_frequency
        eval_steps = trainer_cfg.eval_steps

        is_human_flag = trainer_cfg.human_flag

        eps_start = trainer_cfg.epsilon_max
        eps_end = trainer_cfg.epsilon_min
        eps_decay = trainer_cfg.epsilon_decay

        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start  # initialize epsilon

        loss_window = deque(maxlen=100)
        pos_reward_ratio_window = deque(maxlen=100)
        neg_reward_ratio_window = deque(maxlen=100)

        # start with pre-trained model
        if (model_filename is not None):
            filename = self.select_model_filename(model_filename=model_filename)
            agent.current_model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
            agent.target_model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
            eps = 0.78

        epoch_recorder = Recorder(
            header=['epoch', 'avg_score', 'avg_val_score', 'epsilon', 'avg_loss', 'beta'],
            session_id=self.__session_id,
            experiments_path=self.__config.get_app_experiments_path(train_mode=True),
            model=None,
            log_prefix='epoch-',
            configuration=self.__config.get_current_exp_cfg()
        )

        episode_recorder = Recorder(
            header=['step', 'episode', 'epoch', 'epoch step', 'epoch_episode', 'episode step', 'score', 'epsilon',
                    'beta',
                    'avg_pos_reward_ratio', 'avg_neg_reward_ratio', 'avg_loss'],
            session_id=self.__session_id,
            experiments_path=self.__config.get_app_experiments_path(train_mode=True),
            model=None,
            log_prefix='episode-',
            configuration=self.__config.get_current_exp_cfg()
        )

        EVAL_FREQUENCY = eval_frequency
        EVAL_STEPS = eval_steps
        MAX_STEPS = max_steps
        MAX_EPISODE_STEPS = max_episode_steps

        step = 0
        epoch = 0
        episode = 0

        while step < MAX_STEPS:

            epoch_step = 0

            ################################################################################
            # Training
            ################################################################################

            terminal = True
            epoch_episode = 0

            while (epoch_step < EVAL_FREQUENCY) and (step < MAX_STEPS):

                for episode_step in range(MAX_EPISODE_STEPS):

                    if epoch_step >= EVAL_FREQUENCY:
                        break
                    elif step >= MAX_STEPS:
                        break

                    if terminal:
                        terminal = False

                        state, new_life = env.reset()

                        state = agent.pre_process(state)

                        score = 0
                        epoch_episode += 1

                    action = agent.act(state, eps)

                    if new_life:
                        start_game_action = env.start_game_action()
                        action = start_game_action if start_game_action is not None else action

                    action += env.action_offset()

                    if is_human_flag:
                        env.render(mode='human')

                    next_state, reward, done, new_life = env.step(action)

                    next_state = agent.pre_process(next_state)

                    action -= env.action_offset()

                    pos_reward_ratio, neg_reward_ratio, loss, beta = agent.step(state, action, reward, next_state, done)

                    if loss is not None:
                        loss_window.append(loss)
                        pos_reward_ratio_window.append(pos_reward_ratio)
                        neg_reward_ratio_window.append(neg_reward_ratio)

                    step += 1
                    epoch_step += 1

                    state = next_state
                    score += reward

                    if done:
                        break

                    logging.debug(
                        'Step: {}\tEpisode: {}\tEpoch: {}\tEpoch Step: {}\tEpoch Episode: {}\tEpisode Step: {}\tScore: {:.2f}'
                        '\tEpsilon: {:.2f}\tAvg Pos Reward Ratio: {:.3f}\tAvg Neg Reward Ratio: {:.3f}\tLoss {:.6f}'
                            .format(step, episode, epoch, epoch_step, epoch_episode, episode_step, score, eps,
                                    np.mean(pos_reward_ratio_window) if len(pos_reward_ratio_window) > 0 else 0,
                                    np.mean(neg_reward_ratio_window) if len(neg_reward_ratio_window) > 0 else 0,
                                    np.mean(loss_window) if len(loss_window) > 0 else 0))
                logging.warning(
                    'Step: {}\tEpisode: {}\tEpoch: {}\tEpoch Step: {}\tEpoch Episode: {}\tEpisode Step: {}\tScore: {:.2f}'
                    '\tEpsilon: {:.2f}\tAvg Pos Reward Ratio: {:.3f}\tAvg Neg Reward Ratio: {:.3f}\tLoss {:.6f}'
                        .format(step, episode, epoch, epoch_step, epoch_episode, episode_step, score, eps,
                                np.mean(pos_reward_ratio_window) if len(pos_reward_ratio_window) > 0 else 0,
                                np.mean(neg_reward_ratio_window) if len(neg_reward_ratio_window) > 0 else 0,
                                np.mean(loss_window) if len(loss_window) > 0 else 0))

                episode_recorder.record(
                    [step, episode, epoch, epoch_step, epoch_episode, episode_step, score, eps, beta,
                     np.mean(pos_reward_ratio_window) if len(pos_reward_ratio_window) > 0 else 0,
                     np.mean(neg_reward_ratio_window) if len(neg_reward_ratio_window) > 0 else 0,
                     np.mean(loss_window) if len(loss_window) > 0 else 0])

                episode += 1

                if step <= MAX_STEPS:
                    scores_window.append(score)  # save most recent score

                eps = max(eps_end, eps_decay * eps)  # decrease epsilon

                # sys.stdout.flush()

                episode_recorder.save()

                terminal = True

            ################################################################################
            # Validation
            ################################################################################

            val_step = 0

            val_scores_window = deque(maxlen=100)  # last 100 scores

            terminal = True
            epoch_val_episode = 0

            while val_step < EVAL_STEPS:

                for episode_val_step in range(MAX_EPISODE_STEPS):

                    if val_step >= EVAL_STEPS:
                        break

                    if terminal:
                        terminal = False

                        state, new_life = env.reset()
                        state = agent.pre_process(state)
                        score = 0
                        epoch_val_episode += 1

                    action = agent.act(state, eps)

                    if new_life:
                        start_game_action = env.start_game_action()
                        action = start_game_action if start_game_action is not None else action

                    action += env.action_offset()

                    if is_human_flag:
                        env.render(mode='human')

                    next_state, reward, done, new_life = env.step(action)

                    next_state = agent.pre_process(next_state)

                    val_step += 1

                    state = next_state
                    score += reward

                    if done:
                        break

                    logging.debug(
                        'Epoch: {}\tVal Step: {}\tEpoch Val Episode: {}\tEpisode Step: {}\tVal Score: {:.2f}\tEpsilon: {:.2f}'
                            .format(epoch, val_step, epoch_val_episode, episode_val_step, score, eps))

                logging.warning(
                    'Epoch: {}\tVal Step: {}\tEpoch Val Episode: {}\tEpisode Step: {}\tVal Score: {:.2f}\tEpsilon: {:.2f}'
                        .format(epoch, val_step, epoch_val_episode, episode_val_step, score, eps))

                if val_step < EVAL_STEPS:
                    val_scores_window.append(score)  # save most recent score

                sys.stdout.flush()

                terminal = True

            logging.critical(
                'Epoch {}\t Score: {:.2f}\t Val Score: {:.2f}\tEpsilon: {:.2f}'.format(epoch, np.mean(scores_window),
                                                                                       np.mean(val_scores_window),
                                                                                       eps))

            epoch_recorder.record(
                [epoch, np.mean(scores_window), np.mean(val_scores_window), eps, np.mean(loss_window), beta])
            epoch_recorder.save()

            model_filename = self.get_model_filename(epoch, np.mean(scores_window), np.mean(val_scores_window), eps)

            torch.save(agent.current_model.state_dict(), model_filename)

            epoch += 1

        env.close()

        return scores_window
