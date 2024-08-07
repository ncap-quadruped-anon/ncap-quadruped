"""Estimates base velocity for A1 robot from accelerometer readings."""
import numpy as np

from filterpy.kalman import KalmanFilter
from ncap_deploy.utilities.moving_window_filter import MovingWindowFilter


class VelocityEstimator:
    """Estimates base velocity of A1 robot.

  The velocity estimator consists of 2 parts:
  1) A state estimator for CoM velocity.

  Two sources of information are used:
  The integrated reading of accelerometer and the velocity estimation from
  contact legs. The readings are fused together using a Kalman Filter.

  2) A moving average filter to smooth out velocity readings
  """
    def __init__(self,
                 robot,
                 accelerometer_variance=0.1,
                 sensor_variance=0.1,
                 initial_variance=0.1,
                 moving_window_filter_size=1):
        """Initiates the velocity estimator.

    See filterpy documentation in the link below for more details.
    https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

    Args:
      robot: the robot class for velocity estimation.
      accelerometer_variance: noise estimation for accelerometer reading.
      sensor_variance: noise estimation for motor velocity reading.
      initial_covariance: covariance estimation of initial state.
    """
        self.robot = robot

        self.filter = KalmanFilter(dim_x=3, dim_z=3, dim_u=3)
        self.filter.x = np.zeros(3)
        self._initial_variance = initial_variance
        self.filter.P = np.eye(3) * self._initial_variance  # State covariance
        self.filter.Q = np.eye(3) * accelerometer_variance
        self.filter.R = np.eye(3) * sensor_variance

        self.filter.H = np.eye(3)  # measurement function (y=H*x)
        self.filter.F = np.eye(3)  # state transition matrix
        self.filter.B = np.eye(3)

        self._window_size = moving_window_filter_size
        self.moving_window_filter_x = MovingWindowFilter(
            window_size=self._window_size)
        self.moving_window_filter_y = MovingWindowFilter(
            window_size=self._window_size)
        self.moving_window_filter_z = MovingWindowFilter(
            window_size=self._window_size)
        self._estimated_velocity = np.zeros(3)
        self._last_timestamp = 0

    def reset(self):
        print("resetting velocity estimator")
        self.filter.x = np.zeros(3)
        self.filter.P = np.eye(3) * self._initial_variance
        self.moving_window_filter_x = MovingWindowFilter(
            window_size=self._window_size)
        self.moving_window_filter_y = MovingWindowFilter(
            window_size=self._window_size)
        self.moving_window_filter_z = MovingWindowFilter(
            window_size=self._window_size)
        self._last_timestamp = 0

    def _compute_delta_time(self, current_time):
        if self._last_timestamp == 0.:
            # First timestamp received, return an estimated delta_time.
            delta_time_s = self.robot.time_step
        else:
            delta_time_s = current_time - self._last_timestamp
        self._last_timestamp = current_time
        return delta_time_s

    def update(self, current_time):
        """Propagate current state estimate with new accelerometer reading."""
        delta_time_s = self._compute_delta_time(current_time)
        sensor_acc = self.robot.GetBaseAcceleration()
        self._raw_acc = sensor_acc
        base_orientation = self.robot.GetBaseOrientation()
        rot_mat = self.robot.pybullet_client.getMatrixFromQuaternion(
            base_orientation)
        rot_mat = np.array(rot_mat).reshape((3, 3))

        calibrated_acc = sensor_acc + np.linalg.inv(rot_mat).dot(
            np.array([0., 0., -9.8]))
        self._calibrated_acc = calibrated_acc
        self.filter.predict(u=calibrated_acc * delta_time_s)

        # Correct estimation using contact legs
        observed_velocities = []
        self._observed_velocities = []
        foot_contact = self.robot.GetFootContacts()
        for leg_id in range(4):
            if foot_contact[leg_id]:
                jacobian = self.robot.ComputeJacobian(leg_id)
                # Only pick the jacobian related to joint motors
                joint_velocities = self.robot.GetMotorVelocities()[leg_id *
                                                                   3:(leg_id +
                                                                      1) * 3]
                leg_velocity_in_base_frame = jacobian.dot(joint_velocities)
                base_velocity_in_base_frame = -leg_velocity_in_base_frame[:3]
                observed_velocities.append(base_velocity_in_base_frame)
                self._observed_velocities.append(base_velocity_in_base_frame)
            else:
                self._observed_velocities.append(np.zeros(3))

        if len(observed_velocities) > 0:
            observed_velocities = np.mean(observed_velocities, axis=0)
            self.filter.update(observed_velocities)

        vel_x = self.moving_window_filter_x.calculate_average(self.filter.x[0])
        vel_y = self.moving_window_filter_y.calculate_average(self.filter.x[1])
        vel_z = self.moving_window_filter_z.calculate_average(self.filter.x[2])
        self._estimated_velocity = np.array([vel_x, vel_y, vel_z])

    @property
    def estimated_velocity(self):
        return self._estimated_velocity.copy()
