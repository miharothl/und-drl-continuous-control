import logging

from drl.agents.dqn_agent import DqnAgent
from drl.experiment.config import Config


class AgentFactory:

    def __init__(self):
        pass

    @staticmethod
    def create(config: Config) -> DqnAgent:

        action_size = config.get_current_exp_cfg().agent_cfg.action_size
        state_size = config.get_current_exp_cfg().agent_cfg.state_size
        state_rgb = config.get_current_exp_cfg().agent_cfg.state_rgb
        num_frames = config.get_current_exp_cfg().agent_cfg.num_frames

        logging.debug("Agent action size: {}".format(action_size))
        logging.debug("Agent state size: {}".format(state_size))
        logging.debug("Agent state RGB: {}".format(state_rgb))

        # if state_rgb:
        #     agent = DqnAgentRgb(state_size=state_size, action_size=action_size, seed=None, num_frames=num_frames)
        # else:
        agent = DqnAgent(seed=0, cfg=config)

        return agent
