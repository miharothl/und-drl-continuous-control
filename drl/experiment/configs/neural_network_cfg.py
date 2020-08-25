from typing import List

from drl.experiment.configs.config_base import ConfigBase


class NeuralNetworkConfig(ConfigBase):
    def __init__(self,
                 hidden_layers: List[int],
                 ):
        self.hidden_layers = hidden_layers
