import typing as T
import dataclasses
import os

import hydra

import tonic
import gym
from . import utils


@dataclasses.dataclass
class PlayOutput:
    agent: tonic.agents.Agent
    environment: gym.core.Env
    test_environment: gym.core.Env
    checkpoint_config: dict[str, T.Any] | None
    checkpoint_path: str | None


def play(
    *,
    header: str | None = None,
    agent: T.Mapping | None = None,
    environment: T.Mapping | None = None,
    test_environment: T.Mapping | None = None,
    seed: int = 0,
    checkpoint_output_dir: os.PathLike | str | None = None,
    checkpoint_id: T.Literal['last', 'first', 'none'] | int = 'last',
) -> PlayOutput:
    # Load checkpoint, if specified.
    checkpoint_config = None
    checkpoint_path = None
    if checkpoint_output_dir:
        checkpoint_config, checkpoint_path = utils.load(
            checkpoint_output_dir,
            checkpoint_id='none' if agent is not None else checkpoint_id,
        )
        header = header or checkpoint_config['header']
        agent = agent or checkpoint_config['agent']
        environment = environment or checkpoint_config['environment']
        test_environment = test_environment or checkpoint_config['test_environment']

    # Run the header code (e.g. import frameworks, register environments).
    if header:
        exec(header)

    # Build the training environment.
    environment_cfg = environment
    assert environment_cfg is not None
    environment = hydra.utils.instantiate(environment_cfg)
    environment.seed(seed)

    # Build the testing environment.
    test_environment_cfg = test_environment or environment_cfg
    assert test_environment_cfg is not None
    test_environment = hydra.utils.instantiate(test_environment_cfg)
    test_environment.seed(seed + 10000)

    # Build the agent.
    agent_cfg = agent
    assert agent_cfg is not None
    agent = hydra.utils.instantiate(agent_cfg)
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=seed,
    )
    if checkpoint_path: agent.load(checkpoint_path)

    return PlayOutput(
        agent=agent,
        environment=environment,
        test_environment=test_environment,
        checkpoint_config=checkpoint_config,
        checkpoint_path=checkpoint_path,
    )

__all__ = ['play', 'PlayOutput']