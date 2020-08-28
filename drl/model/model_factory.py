import torch
from typing import Tuple

from drl.experiment.configuration import Configuration
from drl.model.model_dqn import DqnDueling2Hidden, Dqn2Hidden, Dqn3Hidden, Dqn4Hidden
from drl.model.model_dqn_rgb import QNetwork2a


class ModelFactory:

    def __init__(self):
        pass

    @staticmethod
    def create(seed, device, cfg: Configuration) -> Tuple[torch.nn.Module, torch.nn.Module]:

        if cfg.get_current_exp_cfg().agent_cfg.state_rgb is True:
            network_type = 'rgb'
        else:
            network_type = 'classic'

        dueling = False
        if cfg.get_current_exp_cfg().reinforcement_learning_cfg.algorithm_type == 'dqn_dueling':
            dueling = True

        fc_units = cfg.get_current_exp_cfg().neural_network_cfg.hidden_layers
        num_frames = cfg.get_current_exp_cfg().agent_cfg.num_frames
        state_size = cfg.get_current_exp_cfg().agent_cfg.state_size
        action_size = cfg.get_current_exp_cfg().agent_cfg.action_size
        seed = seed
        device = device

        if network_type == "classic":

            if len(fc_units) == 2:

                if dueling:
                    current_model = DqnDueling2Hidden(
                        state_size * num_frames, action_size, seed,
                        fc1_units=fc_units[0],
                        fc2_units=fc_units[1]
                    ).to(device)
                    target_model = DqnDueling2Hidden(
                        state_size * num_frames, action_size, seed,
                        fc1_units=fc_units[0],
                        fc2_units=fc_units[1]
                    ).to(device)
                else:
                    current_model = Dqn2Hidden(
                        state_size * num_frames, action_size, seed,
                        fc1_units=fc_units[0],
                        fc2_units=fc_units[1]
                    ).to(device)
                    target_model = Dqn2Hidden(
                        state_size * num_frames, action_size, seed,
                        fc1_units=fc_units[0],
                        fc2_units=fc_units[1]
                    ).to(device)
            elif len(fc_units) == 3:
                current_model = Dqn3Hidden(
                    state_size * num_frames, action_size, seed,
                    fc1_units=fc_units[0],
                    fc2_units=fc_units[1],
                    fc3_units=fc_units[2]).to(device)

                target_model = Dqn3Hidden(
                    state_size * num_frames, action_size, seed,
                    fc1_units=fc_units[0],
                    fc2_units=fc_units[1],
                    fc3_units=fc_units[2]).to(device)
            elif len(fc_units) == 4:
                current_model = Dqn4Hidden(
                    state_size * num_frames, action_size, seed,
                    fc1_units=fc_units[0],
                    fc2_units=fc_units[1],
                    fc3_units=fc_units[2],
                    fc4_units=fc_units[3]).to(device)

                target_model = Dqn4Hidden(
                    state_size * num_frames, action_size, seed,
                    fc1_units=fc_units[0],
                    fc2_units=fc_units[1],
                    fc3_units=fc_units[2],
                    fc4_units=fc_units[3]).to(device)

            return current_model, target_model

        if type == 'rgb':
            current_model = QNetwork2a(state_size[0], state_size[1], num_frames, action_size, seed).to(device)
            target_model = QNetwork2a(state_size[0], state_size[1], num_frames, action_size, seed).to(device)

            return current_model, target_model
