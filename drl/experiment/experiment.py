import gym
import logging

from gym.spaces import Discrete
from datetime import datetime

from unityagents import UnityEnvironment

from drl.agents.classic.dqn_agent import DqnAgent
from drl.agents.rgb.dqn_agent_rgb import DqnAgentRgb
from drl.environments.gym_atari_env import GymAtariEnv
from drl.environments.gym_standard_env import GymStandardEnv
from drl.environments.unity_env import UnityEnv
from drl.experiment.config import Config
from drl.experiment.player import Player
from drl.experiment.trainer import Trainer

class Experiment:
    def __init__(self, config: Config):
        self.__config = config
        self.__timestamp = datetime.now().strftime("%Y%m%dT%H%M")

    def play(self, mode, model, num_episodes=3, trained=True, num_steps=None):

        with Player(
                   env=self.create_env(),
                   agent=self.create_agent(),
                   config=self.__config,
                   session_id=self.get_session_id()) as player:

            player.play(trained=trained,
                        mode=mode,
                        is_rgb=self.__config.get_current_exp_cfg().agent_cfg.state_rgb,
                        model_filename=model,
                        num_episodes=num_episodes,
                        num_steps=num_steps)

    def play_dummy(self, mode, model, num_episodes=3, num_steps=None):
        self.play(trained=False,
                  mode=mode,
                  model=model,
                  num_episodes=num_episodes,
                  num_steps=num_steps)

    def train(self, model=None):

        trainer = Trainer(
            config=self.__config,
            session_id=self.get_session_id()
        )

        return trainer.train(
            self.create_agent(),
            self.create_env(),
            model_filename=model
        )

    def set_env(self, env):
        self.__config.set_current_exp_cfg(env)

    def create_agent(self):
        action_size = self.__config.get_current_exp_cfg().agent_cfg.action_size
        state_size = self.__config.get_current_exp_cfg().agent_cfg.state_size
        state_rgb = self.__config.get_current_exp_cfg().agent_cfg.state_rgb
        num_frames = self.__config.get_current_exp_cfg().agent_cfg.num_frames

        logging.debug("Agent action size: {}".format(action_size))
        logging.debug("Agent state size: {}".format(state_size))
        logging.debug("Agent state RGB: {}".format(state_rgb))

        if state_rgb:
            agent = DqnAgentRgb(state_size=state_size, action_size=action_size, seed=None, num_frames=num_frames)
        else:
            agent = DqnAgent(seed=0, cfg=self.__config)

        return agent

    def create_env(self):
        env_name = self.__config.get_current_exp_cfg().gym_id

        env_type = self.__config.get_current_exp_cfg().environment_cfg.env_type

        if env_type == 'gym':
            env = GymStandardEnv(name=env_name)
        elif env_type == 'gym_atari':
            env = GymAtariEnv(name=env_name)
        elif env_type == 'unity':
            env = UnityEnv(name=env_name)
        else:
            raise Exception("Environment '{}' type not supported".format(env_type))

        return env

    def list_envs(self):
        envs = self.__config.get_exp_ids()
        for e in envs:
            print(e)

        return envs

    def get_timestamp(self):
        return self.__timestamp

    def get_session_id(self):
        return "{}-{}".format(
            self.__config.get_current_exp_cfg().id,
            self.get_timestamp()
        )
