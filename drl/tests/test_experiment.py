import pytest

from drl.experiment.config2 import Config2
from drl.experiment.experiment import Experiment


class TestExperiment:

    def test_listEnvs_configExist_returnsEnvs(self):

        config = self.__get_config()
        experiment = Experiment(config)

        envs = experiment.list_envs()

        assert len(envs) > 1

    @pytest.mark.depends(name='test_play')
    def test_playDummy_configExist_playsWithDummyAgent(self):
        config = self.__get_config()
        experiment = Experiment(config)

        envs = experiment.list_envs()

        for env in envs:
            experiment.set_env(env)

            if config.get_current_exp_cfg().environment_cfg.env_type == 'unity':
                break

            experiment.play_dummy(mode='rgb-array', model=None, num_episodes=3, num_steps=10)

    @pytest.mark.depends(name='test_train')
    def test_train_configExist_canTrain1Episode(self):
        config = self.__get_config()
        experiment = Experiment(config)

        envs = experiment.list_envs()

        for env in envs:
            experiment.set_env(env)

            if config.get_current_exp_cfg().environment_cfg.env_type == 'unity':
                break

            scores = experiment.train()

            assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)

    @pytest.mark.depends(name='test_train')
    def test_train_unityCconfigExist_canTrain1Episode(self):
        config = self.__get_config()
        experiment = Experiment(config)

        envs = experiment.list_envs()

        for env in envs:
            experiment.set_env(env)

            if config.get_current_exp_cfg().environment_cfg.env_type != 'unity':
                break

            scores = experiment.train()

            assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)


    def __get_config(self):
        cfg = \
            {
                "experiment_cfgs": [
                    {
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
                        "id": "lunarlander",
                        "gym_id": "LunarLander-v2",
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
                        "id": "lunarlander",
                        "gym_id": "LunarLander-v2",
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
                ]
            }

        return Config2(test_flag=True, exp_cfg=cfg)
