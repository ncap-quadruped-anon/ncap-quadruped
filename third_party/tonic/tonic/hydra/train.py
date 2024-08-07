import typing as T
import dataclasses

import omegaconf as oc
import hydra

import tonic
from . import utils


@dataclasses.dataclass
class TrainConfig:
    header: str | None = None
    agent: T.Mapping | None = None
    environment: T.Mapping | None = None
    test_environment: T.Mapping | None = None
    trainer: T.Mapping | None = None
    before_training: str | None = None
    after_training: str | None = None
    parallel: int = 1
    sequential: int = 1
    seed: int = 0
    output_dir: str | None = None
    checkpoint_output_dir: str | None = None
    checkpoint_id: T.Literal['last', 'first', 'none'] | int = 'none'


def train(
    *,
    header: str | None = None,
    agent: T.Mapping | None = None,
    environment: T.Mapping | None = None,
    test_environment: T.Mapping | None = None,
    trainer: T.Mapping | None = None,
    before_training: str | None = None,
    after_training: str | None = None,
    parallel: int = 1,
    sequential: int = 1,
    seed: int = 0,
    output_dir: str | None = None,
    checkpoint_output_dir: str | None = None,
    checkpoint_id: T.Literal['last', 'first', 'none'] | int = 'none',
):
    # Create config to save.
    config = dict(
        header=header,
        agent=agent,
        environment=environment,
        test_environment=test_environment,
        trainer=trainer,
        before_training=before_training,
        after_training=after_training,
        parallel=parallel,
        sequential=sequential,
        seed=seed,
        output_dir=output_dir,
        checkpoint_output_dir=checkpoint_output_dir,
        checkpoint_id=checkpoint_id,
    )

    # Load checkpoint, if specified.
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
        trainer = trainer or checkpoint_config['trainer']

    # Run the header code (e.g. import frameworks, register environments).
    if header:
        exec(header)

    # Build the training environment.
    environment_cfg = environment
    assert environment_cfg is not None
    environment = tonic.environments.distribute(
        lambda: hydra.utils.instantiate(environment_cfg), parallel, sequential)
    environment.initialize(seed=seed)

    # Build the testing environment.
    test_environment_cfg = test_environment or environment_cfg
    assert test_environment_cfg is not None
    test_environment = tonic.environments.distribute(
        lambda: hydra.utils.instantiate(test_environment_cfg))
    test_environment.initialize(seed=seed + 10000)

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

    # Build the trainer.
    trainer_cfg = trainer
    trainer = hydra.utils.instantiate(trainer_cfg)
    trainer.initialize(
        agent=agent,
        environment=environment,
        test_environment=test_environment,
    )

    # Initialize the logger to save data to the output directory.
    output_dir = output_dir or hydra.utils.HydraConfig.get()['runtime']['output_dir']
    tonic.logger.initialize(output_dir, script_path=__file__, config=config)

    # Run some code before training.
    if before_training:
        exec(before_training)

    # Train.
    trainer.run()

    # Run some code after training.
    if after_training:
        exec(after_training)

    return config


# Define simple CLI entrypoint.
@hydra.main(version_base='1.3', config_path='configs', config_name='train')
def main(cfg: oc.DictConfig):
    tonic_cfg = T.cast(dict, oc.OmegaConf.to_container(cfg.tonic, resolve=True))
    train(**tonic_cfg)


if __name__ == '__main__':
    main()

__all__ = ['train', 'TrainConfig']