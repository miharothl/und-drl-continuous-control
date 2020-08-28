import logging

from drl.agent.dqn_agent import DqnAgent
from drl.agent.i_agent import IAgent
from drl.experiment.configuration import Configuration


class AgentFactory:

    def __init__(self):
        pass

    @staticmethod
    def create(config: Configuration) -> IAgent:

        action_size = config.get_current_exp_cfg().agent_cfg.action_size
        state_size = config.get_current_exp_cfg().agent_cfg.state_size
        state_rgb = config.get_current_exp_cfg().agent_cfg.state_rgb

        logging.debug("Agent action size: {}".format(action_size))
        logging.debug("Agent state size: {}".format(state_size))
        logging.debug("Agent state RGB: {}".format(state_rgb))

        agent = DqnAgent(seed=0, cfg=config)

        return agent
