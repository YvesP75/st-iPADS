import gymnasium as gym
from params import *


class TwoDimEnv(gym.Env):
    """
  Custom 2D-Environment that follows gym interface.
  This is a simple 2D-env where the agent must go to the 0,0 point,
  it can go right or straight in a continuous action space, but cannot change speed
  """
    def __init__(self):
        super(TwoDimEnv, self).__init__()

        # agent cannot be further away from target than space_limits
        self.space_limits = SPACE_LIMITS

        # Initialize the agent position
        rho_init = np.random.random() * SPACE_LIMITS
        theta_init = np.random.random() * 2 * PI - PI
        self.rho_init, self.theta_init, self.z_init = rho_init, theta_init, Z_INIT
        self.agent_pos = rho_init * np.exp(1j * theta_init)
        self.agent_z = Z_INIT

        # agent abs speed is constant, may change of direction on the plane, but doesnt change in the z-axis
        self.agent_speed = SPEED_RHO * np.exp(1j*SPEED_ANGLE)
        self.agent_speed_z = SPEED_Z
        self.agent_max_angle = MAX_ANGLE

        #
        self.agent_previous_pos = self.agent_pos


    def reset(self, training=True, rho_init=RHO_INIT, theta_init=THETA_INIT, z_init=Z_INIT):
        """
    :input: TwoDimEnv
    :return: TwoDimEnv
    """
        # Initialize the agent position and speed
        self.agent_pos = rho_init * np.exp(1j * theta_init)
        self.agent_speed = SPEED_RHO * np.exp(1j * SPEED_ANGLE)
        self.agent_z = z_init

        # init step index
        self.step_index = 0

        return self.get_obs()

    def step(self, action):
        '''
        the Gym step
        :param action:
        :return:
        '''

        self.step_index += 1

        # Agent changes of direction according to its action: action[0] in [-1; 1]
        self.agent_speed *= np.exp(1j * self.agent_max_angle * action[0], dtype=complex)
        self.agent_previous_pos = self.agent_pos
        self.agent_pos += self.agent_speed
        self.agent_z -= self.agent_speed_z

        # Are we done?
        done = bool(self.agent_z <= 0)

        # get normalized obs
        obs = self.get_obs()

        return obs, 0., done, {}

    def render(self, **kwargs):
        pass

    def close(self):
        pass

    # calculates normalized obs
    def get_obs(self):
        '''
        normalises the observation
        :return: normalised observation
        '''
        agent_dist = np.abs(self.agent_pos) / self.space_limits
        agent_angle = np.angle(self.agent_pos) / (2 * PI)
        if agent_angle < 0:
            agent_angle += 1
        agent_speed = np.angle(self.agent_speed) / (2 * PI)
        if agent_speed < 0:
            agent_speed += 1
        agent_z = (Z_INIT-self.agent_z) / Z_INIT
        obs = np.array([agent_dist, agent_angle, agent_speed, agent_z]).astype(np.float32)
        return obs


