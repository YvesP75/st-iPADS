from params import *
import gym


class SequenceWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        super(SequenceWrapper, self).__init__(env)
        self.env.max_steps = EPISODE_LENGTH // 10
        self.count = 0

    def reset(self, **kwargs):
        """
        Reset the environment, but grows progressively the number of steps
        """
        self.count += 1
        if self.count % (TRAINING_LENGTH // 10) == 0:
            self.env.max_steps += EPISODE_LENGTH // 10
            print(self.count, self.env.max_steps)

        return self.env.reset(**kwargs)

    def step(self, action):
        """
        :param action: Action taken by the agent
        :return: observation, reward, is the episode over?, additional informations
        """
        return self.env.step(action)

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        pass

