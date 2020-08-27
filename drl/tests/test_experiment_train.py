import pytest

from drl.experiment.config import Config
from drl.experiment.experiment import Experiment
from drl.tests.test_experiment_config import TestExperimentConfig


class TestExperimentTrain:


    @pytest.mark.depends(name='test_train')
    def test_train_environment_gymai_lunarlander(self):
        config = TestExperimentConfig.get()
        experiment = Experiment(config)

        experiment.set_env('lunarlander')
        scores = experiment.train()
        assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)

    def test_train_environment_atari_breakout(self):
        config = TestExperimentConfig.get()
        experiment = Experiment(config)

        experiment.set_env('breakout')
        scores = experiment.train()
        assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)

    def test_train_environment_atari_breakout(self):
        config = TestExperimentConfig.get()
        experiment = Experiment(config)

        experiment.set_env('breakout')
        scores = experiment.train()
        assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)

    def test_train_environment_unity_banana(self):
        config = TestExperimentConfig.get()
        experiment = Experiment(config)

        experiment.set_env('banana')
        # scores = experiment.train()
        # assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)
