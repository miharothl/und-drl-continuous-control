from drl.experiment.configuration.config_base import ConfigBase


class EnvironmentConfig(ConfigBase):
    def __init__(self,
                 env_type: str,
                 ):

        self.ensure_in_list(param=env_type, valid=['gym', 'atari_gym', 'spaceinvaders_atari_gym', 'udacity'])

        self.env_type = env_type
