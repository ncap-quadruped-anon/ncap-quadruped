from absl import app
from absl import logging
import numpy as np
import time
from tqdm import tqdm
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

from ncap_deploy.robots import a1_robot
from ncap_deploy.robots import robot_config

def maintain_position(robot, duration=5, sleep_time=0.005):
    """ Function to maintain the robot's position by setting zero torques. """
    start_time = time.time()
    zero_torques = np.zeros_like(robot.GetMotorAngles())
    while (time.time() - start_time) < duration:
        robot.Step(zero_torques, robot_config.MotorControlMode.TORQUE)
        time.sleep(sleep_time)

def main(_):
    logging.info("WARNING: this code executes low-level controller on the robot.")
    input("Press enter to continue...")

    # Construct sim env and real robot
    p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = a1_robot.A1Robot(pybullet_client=p, action_repeat=1)
    robot.ReceiveObservation()

    # Maintain initial position
    maintain_position(robot)

    # Command torques directly
    desired_torques = np.array([0., 5., -10.] * 4)  # Example torques
    for t in tqdm(range(300)):
        robot.Step(desired_torques, robot_config.MotorControlMode.TORQUE)
        time.sleep(0.005)
        print(robot.GetMotorTorques())

    robot.Terminate()

if __name__ == '__main__':
    app.run(main)