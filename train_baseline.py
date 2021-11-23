import gym, ray
import ray.rllib.agents.a3c as a3c
from ray.tune.logger import pretty_print
import numpy as np
from env.abr import ABRSimEnv
from gym.spaces import Discrete, Box

class myEnv(gym.Env):
    def __init__(self, env_config):
        # Launch ABR environment
        print('Setting up environment..')
        self.env = ABRSimEnv()
        self.action_space = Discrete(6)
        self.observation_space = Box(low=0, high=np.inf, shape=(19,), dtype=np.float32)

    def reset(self):
        initial_obs, initial_obs_extended = self.env.reset()
        return initial_obs
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

if __name__ == "__main__":
    ray.init()
    config = a3c.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["lr"] = 1e-2
    config["lambda"] = 0.96
    config["gamma"] = 0.96

    trainer = a3c.A2CTrainer(config=config, env=myEnv)
    for i in range(100):
        result = trainer.train()
        print(pretty_print(result))

