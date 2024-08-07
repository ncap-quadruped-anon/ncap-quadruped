'''Environment builders for popular domains.'''

import os
import functools
import collections
import typing as T

import gym
import numpy as np
from gym import spaces
from dm_env import specs 

from tonic.utils import logger


@functools.singledispatch
def spec_to_space(spec):
    """Converts a dm_env spec to a Gym space.
    - specs.Array -> spaces.Box
    - specs.BoundedArray -> spaces.Box
    - specs.DiscreteArray -> spaces.Discrete
    - tuple, list -> spaces.Tuple
    - dict -> spaces.Dict
    """
    raise NotImplementedError(f"Unknown spec: {spec}")


@spec_to_space.register(specs.Array)
def spec_to_space_array(spec):
    return spaces.Box(low=-np.inf, high=np.inf, shape=spec.shape, dtype=spec.dtype)


@spec_to_space.register(specs.BoundedArray)
def spec_to_space_bounded_array(spec):
    return spaces.Box(low=spec.minimum, high=spec.maximum, shape=spec.shape, dtype=spec.dtype)


@spec_to_space.register(specs.DiscreteArray)
def spec_to_space_discrete_array(spec):
    return spaces.Discrete(n=spec.num_values)


@spec_to_space.register(tuple)
@spec_to_space.register(list)
def spec_to_space_tuple(spec):
    return spaces.Tuple(tuple(spec_to_space(subspec) for subspec in spec))


@spec_to_space.register(dict)
def spec_to_space_dict(spec):
    # Need `OrderedDict` to prevent sorting upon initialization.
    return spaces.Dict(collections.OrderedDict([
        (key, spec_to_space(subspec)) for key, subspec in spec.items()
    ]))


class DMEnv(gym.core.Env):
    '''Turns a DMEnv environment into a Gym environment.'''

    def __init__(self, environment, random):
        self.environment = environment
        self.random = random

        # Convert from `dm_env.specs` to `gym.spaces`.
        self.observation_space = spec_to_space(self.environment.observation_spec())
        self.action_space = spec_to_space(self.environment.action_spec())

    def seed(self, seed):
        self.random.seed(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def step(self, action):
        try:
            time_step = self.environment.step(action)
            observation = time_step.observation
            reward = time_step.reward
            discount = time_step.discount
            # Keep only true terminations, not truncations (i.e. timeouts).
            done = time_step.last() and discount == 0.
            self.last_time_step = time_step

        # In case MuJoCo crashed.
        except Exception as e:
            path = logger.get_path()
            os.makedirs(path, exist_ok=True)
            save_path = os.path.join(path, 'crashes.txt')
            error = str(e)
            with open(save_path, 'a') as file:
                file.write(error + '\n')
            logger.error(error)
            observation = self.last_time_step.observation
            reward = 0.
            done = True

        return observation, reward, done, {}

    def reset(self):
        time_step = self.environment.reset()
        self.last_time_step = time_step
        return time_step.observation

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        '''Returns RGB frames from a camera.'''
        assert mode == 'rgb_array'
        return self.environment.physics.render(
            height=height, width=width, camera_id=camera_id)