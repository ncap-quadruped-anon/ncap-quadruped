'''Environment builders for popular domains.'''

import os

import gym.wrappers
import numpy as np

from tonic import environments
from tonic.utils import logger


def gym_environment(*args, **kwargs):
    '''Returns a wrapped Gym environment.'''

    def _builder(*args, **kwargs):
        return gym.make(*args, **kwargs)

    return build_environment(_builder, *args, **kwargs)


def bullet_environment(*args, **kwargs):
    '''Returns a wrapped PyBullet environment.'''

    def _builder(*args, **kwargs):
        import pybullet_envs  # noqa
        return gym.make(*args, **kwargs)

    return build_environment(_builder, *args, **kwargs)


def control_suite_environment(*args, **kwargs):
    '''Returns a wrapped Control Suite environment.'''

    def _builder(name, task_kwargs=None, visualize_reward=True, environment_kwargs=None,flatten=True, wrappers=[], **kwargs):
        from dm_control import suite
        from tonic.environments import adapters
        domain_name, task_name = name.split('-')
        dmenv = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs,
        )
        env = adapters.DMEnv(dmenv, dmenv.task.random, **kwargs)
        if flatten:
            env = environments.wrappers.FlattenObservationAction(env)
        for wrapper in wrappers:
            env = wrapper(env)
        step_limit = getattr(dmenv, '_step_limit', float('inf'))
        return gym.wrappers.TimeLimit(env, step_limit)

    return build_environment(_builder, *args, **kwargs)


def composer_environment(env, *args, **kwargs):
    '''Returns a wrapped Composer environment.'''
    
    dmenv = env
    name = type(dmenv.task).__name__

    def _builder(name, dmenv, flatten=True, wrappers=[], **kwargs):
        from tonic.environments import adapters
        env = adapters.DMEnv(dmenv, dmenv.random_state, **kwargs)
        if flatten:
            env = environments.wrappers.FlattenObservationAction(env)
        for wrapper in wrappers:
            env = wrapper(env)
        # Conform to other enviornments which wrap in time limit.
        time_limit = getattr(dmenv, '_time_limit', float('inf'))
        if time_limit < float('inf'):
            step_limit = np.ceil(time_limit / dmenv.task.control_timestep)
        else:
            step_limit = float('inf')
        return gym.wrappers.TimeLimit(env, step_limit)

    return build_environment(_builder, name, dmenv, *args, **kwargs)


def build_environment(
    builder, name, *args, terminal_timeouts=False, time_feature=False,
    max_episode_steps='default', scaled_actions=True, **kwargs
):
    '''Builds and wrap an environment.
    Time limits can be properly handled with terminal_timeouts=False or
    time_feature=True, see https://arxiv.org/pdf/1712.00378.pdf for more
    details.
    '''

    # Build the environment.
    environment = builder(name, *args, **kwargs)

    # Get the default time limit.
    if max_episode_steps == 'default':
        max_episode_steps = environment._max_episode_steps

    # Remove the TimeLimit wrapper if needed.
    if not terminal_timeouts:
        assert type(environment) == gym.wrappers.TimeLimit, environment
        environment = environment.env

    # Add time as a feature if needed.
    if time_feature:
        environment = environments.wrappers.TimeFeature(
            environment, max_episode_steps)

    # Scale actions from [-1, 1]^n to the true action space if needed.
    if scaled_actions:
        environment = environments.wrappers.ActionRescaler(environment)

    environment.name = name
    environment.max_episode_steps = max_episode_steps

    return environment


# Aliases.
Gym = gym_environment
Bullet = bullet_environment
ControlSuite = control_suite_environment
Composer = composer_environment