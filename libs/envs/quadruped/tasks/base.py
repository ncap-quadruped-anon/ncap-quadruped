import abc
import collections
import typing as T

from dm_control import composer
from dm_control.composer import observation, variation
from dm_control.locomotion.walkers import base as base_walker
from dm_env import specs
from envs.quadruped import arenas

# How long between control timesteps (seconds).
# Must be larger than physics timestep.
DEFAULT_CONTROL_TIMESTEP = 0.03

# How long between physics timesteps (seconds).
# Typically small to prevent divergence/instability.
DEFAULT_PHYSICS_TIMESTEP = 0.001


class BaseTask(composer.Task):
  def __init__(
    self,
    robot: base_walker.Walker,
    arena: composer.Arena | None = None,
    freejoint: bool = True,
    physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
    control_timestep: float = DEFAULT_CONTROL_TIMESTEP,
  ):
    self.robot = robot
    self.arena = arena or arenas.Floor(size=(10, 10))
    if freejoint:
      self.arena.add_free_entity(self.robot)
    else:
      self.arena.attach(self.robot)

    self.mjcf_variator = variation.MJCFVariator()
    self.physics_variator = variation.PhysicsVariator()

    # Subclasses can extend this with additional `composer.Observable`s.
    self._task_observables = collections.OrderedDict()

    self.set_timesteps(physics_timestep=physics_timestep, control_timestep=control_timestep)

  # ------------------------------------------------------------------------------------------------
  # Properties

  @property
  def root_entity(self) -> composer.Entity:
    # Implement `composer.Task.root_entity` abstractproperty.
    return self.arena

  @property
  def task_observables(self) -> collections.OrderedDict[str, observation.observable.Observable]:
    # Override `composer.Task.task_observables` property.
    return self._task_observables

  # ------------------------------------------------------------------------------------------------
  # Callbacks

  def initialize_episode_mjcf(self, random_state):
    self.arena.regenerate(random_state)
    self.mjcf_variator.apply_variations(random_state)

  def initialize_episode(self, physics, random_state):
    self.physics_variator.apply_variations(physics, random_state)

  def before_step(self, physics, action, random_state):
    pass

  def before_substep(self, physics, action, random_state):
    # Apply action in the substep (not step) so that PD control operates at the `physics_timestep`,
    # which is much higher frequency than the agent's `control_timestep`.
    self.robot.apply_action(physics, action, random_state)

  def action_spec(self, physics) -> specs.BoundedArray:
    # Override `composer.Task.action_spec`
    return self.robot.action_spec
