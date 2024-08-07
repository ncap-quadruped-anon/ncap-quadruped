import collections
import typing as T

import numpy as np
import utils.mujoco as utm
from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base as base_walker
from dm_control.utils import rewards, transformations
from envs.quadruped.robots import unitree_a1
from envs.quadruped.tasks import base as base_task


class MoveAtFixedVelocity(base_task.BaseTask):
  def __init__(
    self,
    robot: base_walker.Walker,
    termination_roll: float = 180,
    termination_pitch: float = 180,
    ground_friction: tuple[float, float, float] | variation.Variation | None = None,
    action_history: int = 0,
    include_action_history: bool = False,
    **kwargs,
  ):
    super().__init__(robot, **kwargs)
    assert 0 < termination_roll <= 180
    assert 0 < termination_pitch <= 180
    # Orientation: roll = right down (+wx), pitch = front down (+wy).
    self.termination_roll = np.deg2rad(termination_roll)
    self.termination_pitch = np.deg2rad(termination_pitch)
    self._terminating = False

    # Define variations.
    if ground_friction:
      for geom in self.arena.ground_geoms:
        self.mjcf_variator.bind_attributes(geom, friction=ground_friction)

    # Store action history.
    assert action_history >= 0
    self.action_history = collections.deque(maxlen=action_history)
    self.action_zero = utm.sample_spec(self.robot.action_spec, 'zero')
    for _ in range(self.action_history.maxlen):  # type: ignore
      self.action_history.append(self.action_zero)

    # Define task-specific observables.
    obs = self.task_observables
    obs['action_history'] = observable.Generic(lambda _: list(self.action_history))
    obs['action_history'].enabled = include_action_history

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)
    self.action_history.clear()
    for _ in range(self.action_history.maxlen):  # type: ignore
      self.action_history.append(self.action_zero)
    self._terminating = False

  def before_step(self, physics, action, random_state):
    super().before_step(physics, action, random_state)
    self.action_history.append(action)

  def after_step(self, physics, random_state):
    super().after_step(physics, random_state)
    roll, pitch, _ = transformations.quat_to_euler(
      physics.bind(self.robot.mjcf_model.sensor.framequat).sensordata
    )
    self._terminating = (
      np.abs(roll) > self.termination_roll or np.abs(pitch) > self.termination_pitch
    )

  def should_terminate_episode(self, physics):
    return self._terminating

  def get_discount(self, physics):
    return 0. if self._terminating else 1.


class MoveAtFixedVelocity_WalkInThePark(MoveAtFixedVelocity):
  def __init__(
    self,
    robot: base_walker.Walker,
    target_vx: float = 0.5,  # [m/s]
    reward_vx: float = 1.,
    punish_wz: float = 0.1,
    **kwargs,
  ):
    super().__init__(robot, **kwargs)
    self.target_vx = target_vx
    self.reward_vx = reward_vx
    self.punish_wz = punish_wz

  def get_reward(self, physics):
    # Gather sensor data.
    vx, _, _ = physics.bind(self.robot.mjcf_model.sensor.velocimeter).sensordata
    _, _, wz = physics.bind(self.robot.mjcf_model.sensor.gyro).sensordata
    framequat = physics.bind(self.robot.mjcf_model.sensor.framequat).sensordata
    _, pitch, _ = transformations.quat_to_euler(framequat)

    # Gather target data.
    tvx = self.target_vx

    # Construct reward.
    return self.reward_vx * rewards.tolerance(
      np.cos(pitch) * vx,
      bounds=(tvx, 2 * tvx),
      margin=2 * tvx,
      value_at_margin=0,
      sigmoid='linear',
    ) - self.punish_wz * np.abs(wz)


class MoveAtFixedVelocity_PlanarTrackingWithSmoothing(MoveAtFixedVelocity):
  def __init__(
    self,
    robot: unitree_a1.UnitreeA1,
    *,
    # Task hyperparameters.
    target_vx: float = 0.5,  # [m/s]
    target_vy: float = 0.,  # [m/s]
    target_wz: float = 0.,  # [rad/s]
    target_qpos: list[float] | None = None,  # [rad] * len(robot.observable_joints)
    target_swing: float = 0.5,  # [s]
    # Reward hyperparameters.
    reward_vxy: float = 1.,  # Linear velocity tracking (vxy -> target_vxy; sparse).
    reward_wz: float = 0.5,  # Angular velocity tracking (wz -> target_wz; sparse).
    reward_vx: float = 0.,  # Forward velocity (vx -> target_vx; dense).
    reward_swing: float = 0.,  # Feet swing times (swing -> +inf).
    punish_vz: float = 0.,  # Linear velocity penalty (vz -> 0).
    punish_wxy: float = 0.,  # Angular velocity penalty (wxy -> 0).
    punish_roll: float = 0.,  # Roll orientation penalty (roll -> 0).
    punish_pitch: float = 0.,  # Pitch orientation penalty (pitch -> 0).
    punish_qvel: float = 0.,  # Joint velocity penalty (qvel -> 0).
    punish_qacc: float = 0.,  # Joint acceleration penalty (qacc -> 0).
    punish_qtrq: float = 0.,  # Joint torques penalty (qtrq -> 0).
    punish_qwork: float = 0.,  # Joint work penalty (trq * qvel -> 0).
    punish_qpos: float | list[float] = 0.,  # Joint position penalty (qpos -> target_qpos).
    punish_daction: dict[str, float] | None = None,  # Action rate penalty (a[t] - a[t-1] -> 0).
    **kwargs,
  ):
    if punish_daction: kwargs.setdefault('action_history', 2)
    super().__init__(robot, **kwargs)
    self.robot: unitree_a1.UnitreeA1
    self.target_vx = target_vx
    self.target_vy = target_vy
    self.target_wz = target_wz
    self.target_qpos = (
      np.zeros(len(robot.observable_joints)) if target_qpos is None else np.asarray(target_qpos)
    )
    self.target_swing = target_swing
    assert punish_qpos == 0 or target_qpos is not None
    assert not punish_daction or len(self.action_history) >= 2

    # All linear/angular velocities are relative to the robot's IMU reference frame.
    self.reward_vxy = reward_vxy  # vx = +front/-back; vy = +left/-right
    self.reward_wz = reward_wz  # wz (yaw) = +front-goes-left
    self.reward_vx = reward_vx  # vx = +front/-back
    self.reward_swing = reward_swing
    self.punish_vz = punish_vz  # vz = +up/-down
    self.punish_wxy = punish_wxy  # wx (roll) = +right-goes-down; wy (pitch) = +front-goes-down
    self.punish_roll = punish_roll  #
    self.punish_pitch = punish_pitch
    self.punish_qvel = punish_qvel
    self.punish_qacc = punish_qacc
    self.punish_qtrq = punish_qtrq
    self.punish_qwork = punish_qwork
    self.punish_qpos = np.asarray(punish_qpos)
    self.punish_daction: dict[str, float] | T.Literal[0] = punish_daction or 0

    # Cache mjcf model elements.
    self.velocimeter = robot.mjcf_model.sensor.velocimeter
    self.gyro = robot.mjcf_model.sensor.gyro
    self.framequat = robot.mjcf_model.sensor.framequat
    self.joints = robot.observable_joints
    self.actuators = robot.actuators

  def get_reward(self, physics):
    reward_terms = self.get_reward_terms(physics)
    return sum(reward_terms.values()) * self.control_timestep

  def get_reward_terms(self, physics):
    # Gather sensor data.
    vx, vy, vz = physics.bind(self.velocimeter).sensordata
    wx, wy, wz = physics.bind(self.gyro).sensordata
    roll, pitch, _ = transformations.quat_to_euler(physics.bind(self.framequat).sensordata)
    joints = physics.bind(self.joints)
    qpos = joints.qpos
    qvel = joints.qvel
    qacc = joints.qacc
    actuators = physics.bind(self.actuators)
    qtrq = actuators.force
    actions: collections.deque[dict[str, np.ndarray]] = self.action_history
    swing = self.robot.feet_swing_times

    # Compute errors.
    vx_err = vx - self.target_vx
    vy_err = vy - self.target_vy
    wz_err = wz - self.target_wz
    qpos_err = qpos - self.target_qpos
    swing_err = (swing - self.target_swing) * self.robot.feet_swing_to_stance

    # Calculate reward terms.
    terms = {
      'reward_vxy':
        self.reward_vxy * np.exp(-4 * (vx_err**2 + vy_err**2)),
      'reward_wz':
        self.reward_wz * np.exp(-4 * wz_err**2),
      'reward_vx':
        self.reward_vx * np.clip(vx / (self.target_vx + 1e-3), -1, 1),
      'reward_swing':
        self.reward_swing * np.sum(swing_err),
      'punish_vz':
        -self.punish_vz * vz**2,
      'punish_wxy':
        -self.punish_wxy * (wx**2 + wy**2),
      'punish_roll':
        -self.punish_roll * roll**2,
      'punish_pitch':
        -self.punish_pitch * pitch**2,
      'punish_qvel':
        -self.punish_qvel * np.dot(qvel, qvel),
      'punish_qacc':
        -self.punish_qacc * np.dot(qacc, qacc),
      'punish_qtrq':
        -self.punish_qtrq * np.dot(qtrq, qtrq),
      'punish_qwork':
        -self.punish_qwork * np.dot(qtrq, qvel),
      'punish_qpos':
        -np.dot(self.punish_qpos * qpos_err, qpos_err),
      'punish_daction':
        -(
          self.punish_daction and sum((
            w * np.dot(daction := actions[-1][k] - actions[-2][k], daction) for k,
            w in self.punish_daction.items()
          ))
        ),  # type: ignore
    }

    return terms
