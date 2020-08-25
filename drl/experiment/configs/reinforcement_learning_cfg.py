from drl.experiment.configs.config_base import ConfigBase


class ReinforcementLearningConfig(ConfigBase):
    def __init__(self,
                 algorithm_type: str,
                 ):

        self.ensure_in_list(param=algorithm_type, valid=['dqn', 'dqn_double', 'dqn_dueling', 'ddpg'])

        self.algorithm_type = algorithm_type
