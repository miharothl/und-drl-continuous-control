from drl.experiment.configuration.config_base import ConfigBase


class TrainerConfig(ConfigBase):
    def __init__(self,
                 max_steps,
                 max_episode_steps,
                 eval_frequency,
                 eval_steps,
                 epsilon_max,
                 epsilon_min,
                 epsilon_decay,
                 human_flag,
                 batch_size,
                 update_every,
                 learning_rate,
                 tau,
                 gamma,
                 ):

        self.ensure_betwen_0_and_1(epsilon_max)
        self.ensure_betwen_0_and_1(epsilon_min)
        self.ensure_betwen_0_and_1(epsilon_decay)
        self.ensure_betwen_0_and_1(learning_rate)
        self.ensure_betwen_0_and_1(tau)
        self.ensure_betwen_0_and_1(gamma)

        self.ensure_is_greater(param1=epsilon_max, param2=epsilon_min)

        self.max_steps = max_steps
        self.max_episode_steps = max_episode_steps
        self.eval_frequency = eval_frequency
        self.eval_steps = eval_steps
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.human_flag = human_flag
        self.batch_size = batch_size
        self.update_every = update_every
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
