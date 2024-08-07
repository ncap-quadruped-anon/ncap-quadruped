# NCAP Quadruped

## 1. Conda virtual environment

Setup the `conda` virtual environment to manage depencencies.

Due to the many dependencies, you need to use the `libmamba` solver which is [faster and more memory-efficient](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) than the `classic` solver. Otherwise, you will likely see the "Solving environment: Killed" timeout error.

```bash
conda update -n base conda
conda install -n base conda-libmamba-solver

# Optional: Set as global default solver.
conda config --set solver libmamba
```

Install project dependencies from the configuration YAML file:

```bash
conda env create --file env/conda.yaml --solver libmamba

# Alternative: If already created environment and want to update with new YAML file.
conda env update --file env/conda.yaml --solver libmamba
```

Set up the project environment variables:

```bash
# 1. Manually edit "env/conda-vars.sh" with the correct paths on your machine.
vim env/conda-vars.sh

# 2. Initialize the variables.
conda activate ncap
source env/conda-vars.sh

# 3. Reactivate environment and check variables.
conda deactivate
conda activate ncap
conda env config vars list
```

## 2. Train policy in simulation

In the conda environment, run the training scripts with the desired experiment config, like:

```bash
python projects/ncap/train_evolution.py --config-name <path/to/config>
```

## 3. Deploy policy on physical robot

See [robot deployment readme](projects/ncap_deploy/README.md).