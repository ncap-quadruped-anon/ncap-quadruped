'''Environment wrappers.'''

import gym
import numpy as np


class ActionRescaler(gym.ActionWrapper):
    '''Rescales actions from [-1, 1]^n to the true action space.
    The baseline agents return actions in [-1, 1]^n.'''

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Box)
        super().__init__(env)
        high = np.ones(env.action_space.shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-high, high=high)
        true_low = env.action_space.low
        true_high = env.action_space.high
        assert np.all(np.isfinite(true_low))
        assert np.all(np.isfinite(true_high))
        self.bias = (true_high + true_low) / 2
        self.scale = (true_high - true_low) / 2

    def action(self, action):
        return self.bias + self.scale * np.clip(action, -1, 1)


class TimeFeature(gym.Wrapper):
    '''Adds a notion of time in the observations.
    It can be used in terminal timeout settings to get Markovian MDPs.
    '''

    def __init__(self, env, max_steps, low=-1, high=1):
        super().__init__(env)
        dtype = self.observation_space.dtype
        self.observation_space = gym.spaces.Box(
            low=np.append(self.observation_space.low, low).astype(dtype),
            high=np.append(self.observation_space.high, high).astype(dtype))
        self.max_episode_steps = max_steps
        self.steps = 0
        self.low = low
        self.high = high

    def reset(self, **kwargs):
        self.steps = 0
        observation = self.env.reset(**kwargs)
        observation = np.append(observation, self.low)
        return observation

    def step(self, action):
        assert self.steps < self.max_episode_steps
        observation, reward, done, info = self.env.step(action)
        self.steps += 1
        prop = self.steps / self.max_episode_steps
        v = self.low + (self.high - self.low) * prop
        observation = np.append(observation, v)
        return observation, reward, done, info


class FlattenObservationAction(gym.Wrapper):
    '''Flattens both the `observation_space` and the `action_space`.'''
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.utils.flatten_space(self.env.observation_space)
        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)
        
        # Monkey-patch the spaces with pointer to originals. Useful for model modules that need to
        # work with unflattened observations/actions.
        self.observation_space.original = self.env.observation_space
        self.action_space.original = self.env.action_space

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = gym.spaces.utils.flatten(self.env.observation_space, observation)
        return observation
    
    def step(self, action):
        action = gym.spaces.utils.unflatten(self.env.action_space, action)
        observation, reward, done, info = self.env.step(action)
        observation = gym.spaces.utils.flatten(self.env.observation_space, observation)
        return observation, reward, done, info