
from params import *
from SequenceWrapper import SequenceWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import Monitor
from TwoDimEnv import TwoDimEnv



def test(env):
    _ = env.reset()
    action = [-1.0]

    step = 0
    done = False

    while not done:
        print("Step {}".format(step + 1))
        step += 1
        action[0] *= 0.8
        obs, reward, done, info = env.step(action)
        if done:
            print("end of the game", "reward=", reward)
            break


def run_episode(env, model, **kwargs):
    obs = env.reset(training=False, **kwargs)
    done = False
    step = 0
    while not done:
        step += 1
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print('obs=', obs, 'reward=', reward, 'step=', step)


env = TwoDimEnv()
env = SequenceWrapper(env)
sequence_id = str(np.random.randint(10000))
log_dir = './log/'+sequence_id
tensor_dir = './tensor/'+sequence_id
env = Monitor(env, log_dir)
#env.test()
#check_env(env)

model = SAC('MlpPolicy', env, verbose=0, tensorboard_log=tensor_dir)
model.learn(EPISODE_LENGTH*TRAINING_LENGTH, n_eval_episodes=TRAINING_LENGTH)
run_episode(env, model)

