import gym
from gym.spaces import Discrete

from drl import logging
from drl.environments.gym_atari_env import GymAtariEnv
from drl.environments.i_environment import IEnvironment


class GymAtariSpaceInvadersEnv(GymAtariEnv):

    def action_offset(self):
        return 1

    def start_game_action(self):
        return 0

