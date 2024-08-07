import collections
import functools
import os
import pathlib
import typing as T

import numpy as np
import numpy.typing as npt
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.utils import transformations
from dm_env import specs

# Path to MuJoCo XML file relative to repo root.
UNITREE_A1_XML = (
  os.getenv('REPO_ROOT', '.') /
  pathlib.Path('third_party/mujoco_menagerie/unitree_a1/a1_torque.xml')
)

# Control ranges for joint velocities and stiffness/damping coefficients.
# Control ranges for joint positions can be extracted from the XML.
UNITREE_A1_VEL_MIN = -21.0  # rad/s
UNITREE_A1_VEL_MAX = +21.0  # rad/s
UNITREE_A1_KP_MAX = 100.0
UNITREE_A1_KD_MAX = 10.0

# Joint position ranges for a setpoint+delta action space.
UNITREE_A1_DEFAULT_POS_MIN = (-0.15, -0.60, -2.50) * 2 + (-0.15, -0.30, -2.50) * 2
UNITREE_A1_DEFAULT_POS_MID = (+0.00, +0.40, -1.00) * 2 + (+0.00, +0.70, -1.00) * 2
UNITREE_A1_DEFAULT_POS_MAX = (+0.15, +1.20, -0.92) * 2 + (+0.15, +1.50, -0.92) * 2


class UnitreeA1Observables(base.WalkerObservables):
  # ------------------------------------------------------------------------------------------------
  # Observables
  # -----------
  # From `base.WalkerObservables`:
  #   - joints_pos
  #   - sensors_gyro
  #   - sensors_accelerometer
  #   - sensors_framequat
  #
  # Values corresponding the 12 joints are ordered as:
  # [ FR_hip, FR_thigh, FR_calf,
  #   FL_hip, FL_thigh, FL_calf,
  #   RR_hip, RR_thigh, RR_calf,
  #   RL_hip, RL_thigh, RL_calf, ]

  @composer.observable
  def joints_vel(self):
    return observable.MJCFFeature('qvel', self._entity.observable_joints)

  @composer.observable
  def joints_trq(self):
    return observable.MJCFFeature('force', self._entity.actuators)

  @composer.observable
  def sensors_euler(self):
    return observable.Generic(
      lambda physics: transformations.
      quat_to_euler(physics.bind(self._entity.mjcf_model.sensor.framequat).sensordata)
    )

  @composer.observable
  def sensors_rollpitch(self):
    return observable.Generic(
      lambda physics: transformations.
      quat_to_euler(physics.bind(self._entity.mjcf_model.sensor.framequat).sensordata)[:2]
    )

  @composer.observable
  def sensors_velocimeter(self):
    return observable.MJCFFeature('sensordata', self._entity.mjcf_model.sensor.velocimeter)

  @composer.observable
  def sensors_foot(self):
    return observable.MJCFFeature('sensordata', self._entity.mjcf_model.sensor.touch)

  @composer.observable
  def time(self):
    return observable.Generic(lambda physics: np.array([physics.data.time]))

  # ------------------------------------------------------------------------------------------------
  # Properties

  @property
  def proprioception(self):
    return [
      self.joints_pos,
      self.joints_vel,
      self.joints_trq,
    ] + self._collect_from_attachments('proprioception')

  @property
  def kinematic_sensors(self):
    return [
      self.sensors_gyro,
      self.sensors_accelerometer,
      self.sensors_framequat,
      self.sensors_euler,
      self.time,
    ] + self._collect_from_attachments('kinematic_sensors')

  @property
  def dynamic_sensors(self):
    return [
      self.sensors_foot,
    ] + self._collect_from_attachments('dynamic_sensors')


class ActionTransform(T.Protocol):
  input_min: np.ndarray
  input_max: np.ndarray
  output_min: np.ndarray
  output_max: np.ndarray

  def __call__(self, action: np.ndarray) -> np.ndarray:
    ...


class SetpointDeltaTransform(ActionTransform):
  def __init__(self, min: npt.ArrayLike, mid: npt.ArrayLike, max: npt.ArrayLike):
    self.output_mid, self.output_min, self.output_max = np.broadcast_arrays(mid, min, max)
    self.input_mid = np.zeros_like(self.output_mid)
    self.input_min = -np.ones_like(self.output_min)
    self.input_max = +np.ones_like(self.output_max)

  def __call__(self, action: np.ndarray) -> np.ndarray:
    return ((action < 0) * (self.output_min + (action + 1) * (self.output_mid - self.output_min)) +
            (0 <= action) * (self.output_mid + action * (self.output_max - self.output_mid)))


class TorqueGenerator(T.Protocol):
  def __call__(self, *, physics, action, random_state, joints, actuators) -> np.ndarray:
    ...


JOINT_LIMIT_CURVES = dict(
  linear=lambda z: z,
  parabolic=lambda z: z**2,
)


class JointLimits(TorqueGenerator):
  def __init__(
    self,
    min: npt.ArrayLike | None = None,
    max: npt.ArrayLike | None = None,
    margin: npt.ArrayLike | float = 0.2,
    torque: npt.ArrayLike | float = 10.,
    curve: T.Literal['linear', 'parabolic'] = 'parabolic',
  ):
    # TODO: Allow separate margins/torques for min/max.
    # TODO: Allow some elements of min/max/margin to be infinity.
    assert min is not None or max is not None
    assert curve in JOINT_LIMIT_CURVES
    self.min = np.asarray(min) if min is not None else None
    self.max = np.asarray(max) if max is not None else None
    self.zeros = np.zeros_like(self.min if self.min is not None else self.max)
    self.margin = np.asarray(margin, like=self.zeros)
    self.torque = np.asarray(torque, like=self.zeros)
    self.curve = JOINT_LIMIT_CURVES[curve]
    assert (self.torque >= 0).all()

  def __call__(self, *, physics, action, random_state, joints, actuators) -> np.ndarray:
    out = self.zeros
    if self.min is not None:
      z1 = -np.clip(joints.qpos - (self.min + self.margin), -self.margin, 0) / self.margin
      out = out + self.torque * self.curve(z1)
    if self.max is not None:
      z2 = np.clip(joints.qpos - (self.max - self.margin), 0, self.margin) / self.margin
      out = out - self.torque * self.curve(z2)
    return out


class EkebergMuscles(TorqueGenerator):
  def __call__(self, *, physics, action, random_state, joints, actuators) -> np.ndarray:
    flx = action['flx']
    ext = action['ext']
    return flx - ext


class GravityCompensation(TorqueGenerator):
  def __call__(self, *, physics, action, random_state, joints, actuators) -> np.ndarray:
    return np.array(joints.qfrc_bias, dtype=float)


class UnitreeA1(base.Walker):
  # ------------------------------------------------------------------------------------------------
  # Initialization

  def _build(
    self,
    include_trq: bool = True,
    include_pos: bool = True,
    include_vel: bool = True,
    kp: float | npt.ArrayLike | None = None,
    kd: float | npt.ArrayLike | None = None,
    pos_init: npt.ArrayLike | None = UNITREE_A1_DEFAULT_POS_MID,
    vel_init: npt.ArrayLike | None = None,
    trq_transform: ActionTransform | None = None,
    pos_transform: ActionTransform | None = None,
    vel_transform: ActionTransform | None = None,
    trq_additional: list[TorqueGenerator] = [],
    foot_friction: dict[T.Literal['FR', 'FL', 'RR', 'RL'], str] = {},
    feet_stats: bool = False,
    name: str | None = None,
    model_xml_path: os.PathLike = UNITREE_A1_XML,
  ):
    self.include_trq = include_trq
    self.include_pos = include_pos
    self.include_vel = include_vel
    self.kp = np.asarray(kp) if kp is not None else None
    self.kd = np.asarray(kd) if kd is not None else None

    # Parse MuJoCo XML file.
    self._mjcf_model = mjcf.from_path(model_xml_path)
    if name: self._mjcf_model.model = name

    # Remove elements related to freejoints, which are only allowed as worldbody grandchildren.
    freejoint = mjcf.traversal_utils.get_freejoint(self._mjcf_model.worldbody.body['trunk'])
    if freejoint: freejoint.remove()
    keyframe = self._mjcf_model.keyframe
    if keyframe: keyframe.remove()

    # Cache elements that will be exposed as properties.
    self._root_body = self._mjcf_model.find('body', 'trunk')
    self._joints = self._mjcf_model.find_all('joint')
    self._actuators = self._mjcf_model.find_all('actuator')
    assert all(a.joint == j for a, j in zip(self.actuators, self.joints))

    # Adjust model parameters.
    for foot, friction in foot_friction.items():
      self._root_body.find('geom', f'{foot}_foot').friction = friction  # type: ignore

    # Set initial joints pose for start of each episode.
    n_joints = len(self.joints)
    self.pos_init = np.clip(
      np.asarray(pos_init) if pos_init is not None else np.zeros(n_joints),
      *self.jointslimits,
    )
    self.vel_init = np.clip(
      np.asarray(vel_init) if vel_init is not None else np.zeros(n_joints),
      UNITREE_A1_VEL_MIN,
      UNITREE_A1_VEL_MAX,
    )

    # Register action transforms, which convert from the "input" actions that the agent generates to
    # the "output" actions used in the control law.
    self.trq_transform = trq_transform
    self.pos_transform = pos_transform
    self.vel_transform = vel_transform

    # Register additional torques, which are feedforward terms added to the control torques (e.g.
    # for gravity compensation).
    self.trq_additional = trq_additional

    # Initialize `update_feet` vars.
    self.feet_stats = feet_stats
    self.feet_sensors = self.mjcf_model.sensor.touch
    assert all(f.name.endswith('foot') for f in self.feet_sensors)

  def _build_observables(self):
    observables = UnitreeA1Observables(self)
    observables.enable_all()
    return observables

  # ------------------------------------------------------------------------------------------------
  # Properties

  @property
  def mjcf_model(self):
    # Implement `composer.Entity.mjcf_model` abstractproperty.
    return self._mjcf_model

  @property
  def root_body(self):
    # Implement `locomotion.Walker.root_body` abstractproperty.
    return self._root_body

  @property
  def observable_joints(self):
    # Implement `locomotion.Walker.observable_joints` abstractproperty.
    return self._joints

  @property
  def joints(self):
    # For completeness along with `actuators`.
    return self._joints

  @property
  def actuators(self):
    # Implement `composer.Robot.actuators` abstractproperty.
    return self._actuators

  @functools.cached_property
  def jointslimits(self):
    minimum = [j.dclass.joint.range[0] for j in self.joints]
    maximum = [j.dclass.joint.range[1] for j in self.joints]
    return minimum, maximum

  @functools.cached_property
  def ctrllimits(self):
    minimum = [a.dclass.motor.ctrlrange[0] for a in self.actuators]
    maximum = [a.dclass.motor.ctrlrange[1] for a in self.actuators]
    return minimum, maximum

  @functools.cached_property
  def action_spec(self):
    # Override `locomotion.Walker.action_spec` property.
    spec = collections.OrderedDict()
    names = ','.join([act.name for act in self.actuators])
    if self.include_trq:
      spec['trq'] = specs.BoundedArray(
        shape=(len(self.actuators),),
        dtype=float,
        minimum=(
          self.trq_transform.input_min if self.trq_transform is not None else
          [a.dclass.motor.ctrlrange[0] for a in self.actuators]
        ),
        maximum=(
          self.trq_transform.input_max if self.trq_transform is not None else
          [a.dclass.motor.ctrlrange[1] for a in self.actuators]
        ),
        name=f'trq:{names}',
      )
    if self.include_pos:
      spec['pos'] = specs.BoundedArray(
        shape=(len(self.actuators),),
        dtype=float,
        minimum=(
          self.pos_transform.input_min if self.pos_transform is not None else
          [a.joint.dclass.joint.range[0] for a in self.actuators]
        ),
        maximum=(
          self.pos_transform.input_max if self.pos_transform is not None else
          [a.joint.dclass.joint.range[1] for a in self.actuators]
        ),
        name=f'pos:{names}',
      )
    if self.include_vel:
      spec['vel'] = specs.BoundedArray(
        shape=(len(self.actuators),),
        dtype=float,
        minimum=(
          self.vel_transform.input_min if self.vel_transform is not None else
          [UNITREE_A1_VEL_MIN for a in self.actuators]
        ),
        maximum=(
          self.vel_transform.input_max if self.vel_transform is not None else
          [UNITREE_A1_VEL_MAX for a in self.actuators]
        ),
        name=f'vel:{names}',
      )
    if self.include_pos and self.kp is None:
      spec['kp'] = specs.BoundedArray(
        shape=(len(self.actuators),),
        dtype=float,
        minimum=[0. for a in self.actuators],
        maximum=[UNITREE_A1_KP_MAX for a in self.actuators],
        name=f'kp:{names}',
      )
    if self.include_vel and self.kd is None:
      spec['kd'] = specs.BoundedArray(
        shape=(len(self.actuators),),
        dtype=float,
        minimum=[0. for a in self.actuators],
        maximum=[UNITREE_A1_KD_MAX for a in self.actuators],
        name=f'kd:{names}',
      )
    return spec

  # ------------------------------------------------------------------------------------------------
  # Callbacks

  def initialize_episode(self, physics, random_state):
    joints = physics.bind(self.joints)
    joints.qpos = self.pos_init
    joints.qvel = self.vel_init
    if self.feet_stats: self.init_feet(physics, random_state)

  def after_step(self, physics, random_state):
    if self.feet_stats: self.update_feet(physics, random_state)

  # ------------------------------------------------------------------------------------------------
  # Methods

  def init_feet(self, physics, random_state):
    self.feet_contacts = np.zeros(len(self.feet_sensors), dtype=bool)
    self.feet_stance_to_swing = np.zeros(len(self.feet_sensors), dtype=bool)
    self.feet_swing_to_stance = np.zeros(len(self.feet_sensors), dtype=bool)
    self.feet_swing_times = np.zeros(len(self.feet_sensors))
    self.feet_stance_times = np.zeros(len(self.feet_sensors))
    self.feet_update_time = 0.

  def update_feet(self, physics, random_state):
    feet = physics.bind(self.feet_sensors).sensordata.copy()
    contacts_prev = self.feet_contacts
    contacts_curr = feet > 0
    stance_to_swing = contacts_prev & ~contacts_curr
    swing_to_stance = ~contacts_prev & contacts_curr
    time_prev = self.feet_update_time
    time_curr = physics.time()
    time_elapsed = time_curr - time_prev

    # A given foot's `[swing,stance]_time` is reset whenever the foot transitions between states,
    # i.e. its contact changes. For example, if a foot changes from swing (`contact == False`) to
    # stance (`contact == True`), the `stance_time` will be reset to 0, and the `swing_time` will
    # remain at its last value throughout the stance until the next swing resets it.
    self.feet_contacts = contacts_curr
    self.feet_stance_to_swing = stance_to_swing
    self.feet_swing_to_stance = swing_to_stance
    self.feet_swing_times += ~contacts_curr * time_elapsed
    self.feet_swing_times *= ~stance_to_swing
    self.feet_stance_times += contacts_curr * time_elapsed
    self.feet_stance_times *= ~swing_to_stance
    self.feet_update_time = time_curr

  def apply_action(self, physics, action, random_state):
    # Override `locomotion.Walker.apply_action` method.
    joints = physics.bind(self.joints)
    actuators = physics.bind(self.actuators)

    ctrl = np.zeros(len(self.actuators))

    for trq_gen in self.trq_additional:
      ctrl += trq_gen(
        physics=physics,
        action=action,
        random_state=random_state,
        joints=joints,
        actuators=actuators,
      )

    if self.include_trq:
      trq = action['trq']
      if self.trq_transform:
        trq = self.trq_transform(trq)
      ctrl += trq

    if self.include_pos or self.kp is not None:
      pos = action.get('pos', self.pos_init)
      if self.pos_transform and 'pos' in action:
        pos = self.pos_transform(pos)
      kp = action['kp'] if self.kp is None else self.kp
      ctrl += kp * (pos - joints.qpos)

    if self.include_vel or self.kd is not None:
      vel = action.get('vel', 0.)
      if self.vel_transform and 'vel' in action:
        vel = self.vel_transform(vel)
      kd = action['kd'] if self.kd is None else self.kd
      ctrl += kd * (vel - joints.qvel)

    actuators.ctrl = np.clip(ctrl, *self.ctrllimits)


class UnitreeA1Constrained(UnitreeA1):
  def _build(
    self,
    pos_constraints: T.Sequence[float | None] | None = None,
    trq_constraints: T.Sequence[float | None] | None = None,
    **kwargs,
  ):
    super()._build(**kwargs)

    self.pos_constraints = pos_constraints is not None
    if pos_constraints is not None:
      assert len(self.actuators) == len(pos_constraints)
      cidxs, cvalues = zip(*[(idx, value) for idx, value in enumerate(pos_constraints) if value is not None])
      uidxs = [idx for idx, value in enumerate(pos_constraints) if value is None]
      self.pos_cidxs: list[int] = list(cidxs)
      self.pos_cvalues: list[float] = list(cvalues)
      self.pos_uidxs: list[int] = uidxs

    self.trq_constraints = trq_constraints is not None
    if trq_constraints is not None:
      assert len(self.actuators) == len(trq_constraints)
      cidxs, cvalues = zip(*[(idx, value) for idx, value in enumerate(trq_constraints) if value is not None])
      uidxs = [idx for idx, value in enumerate(trq_constraints) if value is None]
      self.trq_cidxs: list[int] = list(cidxs)
      self.trq_cvalues: list[float] = list(cvalues)
      self.trq_uidxs: list[int] = uidxs

  @functools.cached_property
  def action_spec(self):
    spec = super().action_spec
    if self.include_pos and self.pos_constraints:
      names = ','.join(np.array([act.name for act in self.actuators])[self.pos_uidxs])
      spec['pos'] = specs.BoundedArray(
        shape=(len(self.pos_uidxs),),
        dtype=float,
        minimum=spec['pos'].minimum[self.pos_uidxs],
        maximum=spec['pos'].maximum[self.pos_uidxs],
        name=f'pos:{names}',
      )
    if self.include_trq and self.trq_constraints:
      names = ','.join(np.array([act.name for act in self.actuators])[self.trq_uidxs])
      spec['trq'] = specs.BoundedArray(
        shape=(len(self.trq_uidxs),),
        dtype=float,
        minimum=spec['trq'].minimum[self.trq_uidxs],
        maximum=spec['trq'].maximum[self.trq_uidxs],
        name=f'trq:{names}',
      )
    return spec

  def apply_action(self, physics, action, random_state):
    action = action.copy()
    if self.include_pos and self.pos_constraints:
      pos = np.zeros_like(self.pos_init)
      pos[self.pos_cidxs] = self.pos_cvalues
      pos[self.pos_uidxs] = action['pos']
      action['pos'] = pos
    if self.include_trq and self.trq_constraints:
      trq = np.zeros_like(self.pos_init)
      trq[self.trq_cidxs] = self.trq_cvalues
      trq[self.trq_uidxs] = action['trq']
      action['trq'] = trq

    return super().apply_action(physics, action, random_state)
