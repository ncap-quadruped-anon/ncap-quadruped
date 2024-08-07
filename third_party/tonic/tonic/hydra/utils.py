import os
import yaml
import typing as T
import pathlib

import tonic


def load(
    checkpoint_output_dir: os.PathLike | str,
    checkpoint_id: T.Literal['last', 'first', 'none'] | int = 'none',
):
    tonic.logger.log(f'Loading experiment from {checkpoint_output_dir}')
    
    # Load the agent weights checkpoint.
    if checkpoint_id == 'none':
        # Use no checkpoint.
        tonic.logger.log('Not loading any checkpoint weights')
        checkpoint_path = None
    else:
        checkpoint_path = os.path.join(checkpoint_output_dir, 'checkpoints')
        if not os.path.isdir(checkpoint_path):
            tonic.logger.error(f'{checkpoint_path} is not a directory')
            raise RuntimeError

        # List all the checkpoints.
        checkpoint_ids = []
        for file in os.listdir(checkpoint_path):
            if file[:5] == 'step_':
                checkpoint_ids.append(int(file.split('.')[0][5:]))

        if checkpoint_ids:
            if checkpoint_id == 'last':
                # Use the last checkpoint.
                checkpoint_id = max(checkpoint_ids)
                checkpoint_path = os.path.join(checkpoint_path, f'step_{checkpoint_id}')
            elif checkpoint_id == 'first':
                # Use the first checkpoint.
                checkpoint_id = min(checkpoint_ids)
                checkpoint_path = os.path.join(checkpoint_path, f'step_{checkpoint_id}')
            else:
                # Use the specified checkpoint number.
                checkpoint_id = int(checkpoint_id)
                if checkpoint_id in checkpoint_ids:
                    checkpoint_path = os.path.join(checkpoint_path, f'step_{checkpoint_id}')
                else:
                    tonic.logger.error(f'No checkpoint {checkpoint_id} found in {checkpoint_path}')
                    checkpoint_path = None
        else:
            tonic.logger.error(f'No checkpoints found in {checkpoint_path}')
            checkpoint_path = None

    # Load the experiment configuration.
    with open(os.path.join(checkpoint_output_dir, 'config.yaml'), 'r') as f:
        checkpoint_config = yaml.load(f, Loader=yaml.FullLoader)
    
    return checkpoint_config, checkpoint_path


__all__ = ['load']