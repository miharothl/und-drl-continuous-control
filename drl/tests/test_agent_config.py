import pytest

from drl.experiment.config import Config
from drl.experiment.experiment import Experiment


class TestEnvironmentConfig:

    @staticmethod
    def get():
        cfg = \
            {
                "experiment_cfgs": [
                    {
                        "id": "lunarlander",
                        "gym_id": "LunarLander-v2",
                        "agent_cfg": {
                            "action_size": 4,
                            "discrete": True,
                            "num_frames": 1,
                            "state_rgb": False,
                            "state_size": 8
                        },
                        "environment_cfg": {
                            "env_type": "gym"
                        },
                        "neural_network_cfg": {
                            "hidden_layers": [
                                64,
                                64
                            ]
                        },
                        "reinforcement_learning_cfg": {
                            "algorithm_type": "dqn"
                        },
                        "replay_memory_cfg": {
                            "buffer_size": 100000,
                            "prioritized_replay": True,
                            "prioritized_replay_alpha": 0.6,
                            "prioritized_replay_beta0": 0.4,
                            "prioritized_replay_eps": 1e-06
                        },
                        "trainer_cfg": {
                            "batch_size": 64,
                            "epsilon_decay": 0.995,
                            "epsilon_max": 1,
                            "epsilon_min": 0.01,
                            "eval_frequency": 16,
                            "eval_steps": 4,
                            "gamma": 0.99,
                            "human_flag": False,
                            "learning_rate": 0.0001,
                            "max_episode_steps": 2,
                            "max_steps": 128,
                            "tau": 0.001,
                            "update_every": 4
                        }
                    },
                    {
                        "id": "breakout",
                        "gym_id": "Breakout-ram-v4",
                        "agent_cfg": {
                            "action_size": 3,
                            "discrete": True,
                            "num_frames": 1,
                            "state_rgb": False,
                            "state_size": 128
                        },
                        "environment_cfg": {
                            "env_type": "spaceinvaders_atari_gym"
                        },
                        "neural_network_cfg": {
                            "hidden_layers": [
                                64,
                                64
                            ]
                        },
                        "reinforcement_learning_cfg": {
                            "algorithm_type": "dqn"
                        },
                        "replay_memory_cfg": {
                            "buffer_size": 100000,
                            "prioritized_replay": True,
                            "prioritized_replay_alpha": 0.6,
                            "prioritized_replay_beta0": 0.4,
                            "prioritized_replay_eps": 1e-06
                        },
                        "trainer_cfg": {
                            "batch_size": 64,
                            "epsilon_decay": 0.995,
                            "epsilon_max": 1,
                            "epsilon_min": 0.01,
                            "eval_frequency": 16,
                            "eval_steps": 4,
                            "gamma": 0.99,
                            "human_flag": False,
                            "learning_rate": 0.0001,
                            "max_episode_steps": 2,
                            "max_steps": 128,
                            "tau": 0.001,
                            "update_every": 4
                        }
                    },
                    {
                        "id": "breakout-rgb",
                        "gym_id": "Breakout-v4",
                        "agent_cfg": {
                            "action_size": 3,
                            "discrete": True,
                            "num_frames": 1,
                            "state_rgb": True,
                            "state_size": [80, 80]
                        },
                        "environment_cfg": {
                            "env_type": "spaceinvaders_atari_gym"
                        },
                        "neural_network_cfg": {
                            "hidden_layers": [
                                64,
                                64
                            ]
                        },
                        "reinforcement_learning_cfg": {
                            "algorithm_type": "dqn"
                        },
                        "replay_memory_cfg": {
                            "buffer_size": 100000,
                            "prioritized_replay": True,
                            "prioritized_replay_alpha": 0.6,
                            "prioritized_replay_beta0": 0.4,
                            "prioritized_replay_eps": 1e-06
                        },
                        "trainer_cfg": {
                            "batch_size": 64,
                            "epsilon_decay": 0.995,
                            "epsilon_max": 1,
                            "epsilon_min": 0.01,
                            "eval_frequency": 16,
                            "eval_steps": 4,
                            "gamma": 0.99,
                            "human_flag": False,
                            "learning_rate": 0.0001,
                            "max_episode_steps": 2,
                            "max_steps": 128,
                            "tau": 0.001,
                            "update_every": 4
                        }
                    },
                    {
                        'id': 'banana',
                        'gym_id': 'env/unity/mac/banana',
                        "agent_cfg": {
                            "action_size": 4,
                            "discrete": True,
                            "num_frames": 1,
                            "state_rgb": False,
                            "state_size": 8
                        },
                        "environment_cfg": {
                            "env_type": "gym"
                        },
                        "neural_network_cfg": {
                            "hidden_layers": [
                                64,
                                64
                            ]
                        },
                        "reinforcement_learning_cfg": {
                            "algorithm_type": "dqn_double"
                        },
                        "replay_memory_cfg": {
                            "buffer_size": 100000,
                            "prioritized_replay": True,
                            "prioritized_replay_alpha": 0.6,
                            "prioritized_replay_beta0": 0.4,
                            "prioritized_replay_eps": 1e-06
                        },
                        "trainer_cfg": {
                            "batch_size": 64,
                            "epsilon_decay": 0.995,
                            "epsilon_max": 1,
                            "epsilon_min": 0.01,
                            "eval_frequency": 16,
                            "eval_steps": 4,
                            "gamma": 0.99,
                            "human_flag": False,
                            "learning_rate": 0.0001,
                            "max_episode_steps": 2,
                            "max_steps": 128,
                            "tau": 0.001,
                            "update_every": 4
                        }
                    },
                ]
            }

        return Config(test_flag=True, exp_cfg=cfg)
