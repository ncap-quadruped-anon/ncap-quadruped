'''Script used to train agents.'''
import signal
import typing as T

import dm_control.composer  # pylint: disable=unused-import
import envs.quadruped  # pylint: disable=unused-import
import envs.wrappers  # pylint: disable=unused-import
import hydra
import omegaconf as oc
import tonic
import tonic.hydra

import tools.infra  # pylint: disable=unused-import


@hydra.main(version_base='1.3', config_path='configs')
def main(cfg: oc.DictConfig) -> None:
  # Override SIGTERM handler so spawned processes don't adopt Submitit's bypass behavior.
  # See: https://github.com/facebookincubator/submitit/issues/1677
  signal.signal(signal.SIGTERM, lambda *args, **kwargs: exit(0))

  tonic_cfg = T.cast(dict[str, T.Any], oc.OmegaConf.to_container(cfg.tonic, resolve=True))
  tonic.hydra.train(**tonic_cfg)


if __name__ == '__main__':
  main()
