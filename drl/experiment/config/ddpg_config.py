from drl.experiment.config.config_base import ConfigBase


class DdpgConfig(ConfigBase):
    def __init__(self,
                 lr_actor,
                 lr_critic,
                 weight_decay
                 ):

        self.ensure_betwen_0_and_1(lr_actor)
        self.ensure_betwen_0_and_1(lr_critic)
        self.ensure_betwen_0_and_1(weight_decay)

        self.lr_actor = lr_actor          # learning rate of the actor
        self.lr_critic = lr_critic        # learning rate of the critic
        self.weight_decay = weight_decay  # L2 weight decay
