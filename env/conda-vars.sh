#!/bin/bash -e

# Absolute path to repo root directory (no whitespaces or trailing slash).
# This directory will be the working directory when executing all scripts.
REPO_ROOT="$HOME/ncap-quadruped"

# Absolute path to output root directory (no whitespaces of trailing slash).
# This directory will be where all scripts write outputs via infrastructure like Hydra.
OUTPUT_ROOT="$HOME/ncap-quadruped/outputs"

# Name of the repo conda environment.
CONDA_ENV=ncap

# ==================================================================================================

# Ensure running scripts will place any packages from the libraries (custom and third-party) into
# the top-level namespace (e.g. "import mylib"). Other packages (projects and tools) will need to
# use a namespace package (e.g. "import tools.infra").
PYTHONPATH="$REPO_ROOT/libs"
PYTHONPATH="$PYTHONPATH:$REPO_ROOT/third_party/tonic"
PYTHONPATH="$PYTHONPATH:$REPO_ROOT/third_party"
PYTHONPATH="$PYTHONPATH:$REPO_ROOT"

# Persist environment variables to conda environment config.
conda activate $CONDA_ENV
conda env config vars set REPO_ROOT=$REPO_ROOT
conda env config vars set OUTPUT_ROOT=$OUTPUT_ROOT
conda env config vars set PYTHONPATH=$PYTHONPATH
conda env config vars set MUJOCO_GL=osmesa
conda deactivate
