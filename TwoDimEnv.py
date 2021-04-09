import gym
from gym import spaces
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
        self.landing_limits = LANDING_LIMITS

        # max number of steps
        self.max_steps = EPISODE_LENGTH

        # lets start at 0
        self.step_index = 0

        # Initialize the agent position
        rho_init = np.random.random() * SPACE_LIMITS
        theta_init = np.random.random() * 2 * PI - PI
        self.rho_init, self.theta_init, self.z_init = rho_init, theta_init, Z_INIT
        self.agent_pos = rho_init * np.exp(1j*theta_init)
        self.agent_z = Z_INIT

        # agent abs speed is constant, may change of direction on the plane, but doesnt change in the z-axis
        self.agent_speed = SPEED_RHO * np.exp(1j*SPEED_ANGLE)
        self.agent_speed_z = SPEED_Z
        self.agent_max_angle = MAX_ANGLE

        #
        self.agent_previous_pos = self.agent_pos
        self.reward = 0
        self.rotation = 0
        self.full_rotation = False

        low, high = 0.0, 1.0
        self.observation_space = spaces.Box(low=low, high=high, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-high, high=high, shape=(1,), dtype=np.float32)

    def reset(self, training=True, rho_init=RHO_INIT, theta_init=THETA_INIT, z_init=Z_INIT):
        """
    :input: TwoDimEnv
    :return: TwoDimEnv
    """
        # Initialize the agent
        # agent speed is constant per step
        if training:
            rho_init = np.random.random() * START_LIMITS
            theta_init = np.random.random() * 2 * PI - PI
            z_init = self.max_steps * SPEED_Z

        # Initialize the agent position and speed
        self.agent_pos = rho_init * np.exp(1j * theta_init)
        self.agent_speed = SPEED_RHO * np.exp(1j * SPEED_ANGLE)
        self.agent_z = z_init

        # init step index
        self.step_index = 0

        # The parachute may turn either left or right
        self.rotation = 0
        self.full_rotation = False

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

        # have you already made a full turn?
        self.rotation += action[0] * SPEED_ANGLE
        if 2 * PI < np.abs(self.rotation):
            self.full_rotation = True

        # Are we done?
        done = bool(self.agent_z <= 0)

        # reward is given according to the distance to (0,0): 1 when at target, 0 when at init, and -1 when very far
        # and : no mercy if you hit the space boundaries
        # no mercy, '8-like loops' are not allowed: either you clockwise or counter clockwise, not both
        if done:
            reward = 2 * (np.exp(1-np.abs(self.agent_pos)/self.landing_limits) / np.exp(1)) - 1
        else:
            reward = 0
            # Account for the boundaries of the space
            if self.out_of_bound() or (self.rotation == 0 and self.full_rotation):
                reward = -1
                done = True
        self.reward = reward

        # For further usage
        info = {}

        # get normalized obs
        obs = self.get_obs()

        return obs, reward, done, info

    def render(self, **kwargs):
        print('obs=', self.get_obs(), 'pos=', self.agent_pos, 'z=', self.agent_z)

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

    def out_of_bound(self):
        '''
        checks wether the parachute is out of bound for the training: it is a condition for early termination
        :return: False if the agent is out of bound
        '''
        low_boundary = LANDING_LIMITS + self.agent_z * SPEED_RHO / SPEED_Z
        mid_boundary = SPACE_LIMITS
        # there is no need to calculate a high limit because the PADS is constrained by its start anyway
        return min(low_boundary, mid_boundary) < np.abs(self.agent_pos)

