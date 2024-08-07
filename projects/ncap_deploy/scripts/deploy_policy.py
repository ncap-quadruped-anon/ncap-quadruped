import numpy as np
import os
import pathlib
import time
import logging
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from absl import app
import torch
import projects.ncap.src.evolution.tonic_torch_rollout as ttr
import hydra
import omegaconf
import pybullet
from pybullet_utils import bullet_client
import pybullet_data
from robot_interface import RobotInterface # pylint: disable=import-error

from ncap_deploy.robots import a1_robot, robot_config

os.chdir(os.getenv('REPO_ROOT'))

def get_robot_data(robot):
    current_motor_angles = np.array(robot._motor_angles)
    joint_torques = -np.array(robot._observed_motor_torques)
    joint_velocities = np.array(robot.GetMotorVelocities())
    foot_contacts = np.array(robot._raw_state.footForce)
    # time = robot.

    data = {
        'time': time.time(),
        'joints_pos': current_motor_angles,
        'joints_trq': joint_torques,
        'joints_vel': joint_velocities,
        'sensors_foot': foot_contacts
    }
    return data

def print_colored(label, data, color='black'):
    color_map = {
        'black': '90', 'red': '91', 'green': '92', 'yellow': '93',
        'blue': '94', 'magenta': '95', 'cyan': '96', 'white': '97'
    }

    color_code = color_map.get(color.lower(), '90')  # Default to black if color not found

    with np.printoptions(precision=3, suppress=True, sign=' ', linewidth=500):
        if isinstance(data, list):
            data = np.array(data)
        if isinstance(data, np.ndarray):
            value = str(data)
        elif isinstance(data, (float, int)):
            value = f'{data:.3f}'
        elif isinstance(data, dict):
            value = '{\n' + ',\n'.join([f'  {key:<10s}: {str(value)}' for key, value in data.items()]) + '\n}'
        else:
            value = str(data)
    text = f'{label:<10s}{value}'
    print(f'\033[{color_code}m{text}\033[0m')

class TonicTorchRolloutPolicy:
    def __init__(self, model):
        self.model = model

    def __call__(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return ttr.as_numpy(self.model.actor(ttr.as_tensor(obs)))


job_dir = '<path/to/trained/model'  # TODO: Replace the 'job_dir' with the path to trained model.
cfg = omegaconf.OmegaConf.load(os.path.join(job_dir, '.hydra/config.yaml'))
print('Configs:', cfg)
job_dir = pathlib.Path(job_dir)
checkpoint = ttr.load_checkpoint(job_dir, epoch='last')
print('Checkpoint:', checkpoint)
pos_transform = hydra.utils.instantiate(cfg.robot.pos_transform)

def legs2_to_legs3(legs2: np.ndarray) -> np.ndarray:
    legs3 = np.zeros(12)
    legs3[1:3] = legs2[0:2]
    legs3[4:6] = legs2[2:4]
    legs3[7:9] = legs2[4:6]
    legs3[10:12] = legs2[6:8]
    return legs3

def predict_action(normalized_obs, model):
    with torch.no_grad():
        obs_tensor = ttr.as_tensor(normalized_obs)
        action_tensor = model.actor(obs_tensor)
        action = ttr.as_numpy(action_tensor)
        return action['pos']

def main(_):
    logging.info("WARNING: this code executes low-level controller on the robot.")
    input("Hang Robot on Rack or ensure that this is a tested policy and Press Enter to Continue...")

    p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = a1_robot.A1Robot(pybullet_client=p, action_repeat=1, motor_kp=10, motor_kd=cfg.robot.kd)
    robot.ReceiveObservation()

  # Move the motors slowly to initial position
    current_motor_angle = np.array(robot.GetMotorAngles())
    mid_pose = np.array([0., 0.9, -1.8] * 4)
    for t in tqdm(range(300)):
        blend_ratio = np.minimum(t / 200., 1)
        action = (1 - blend_ratio) * current_motor_angle + blend_ratio * mid_pose
        robot.Step(action, robot_config.MotorControlMode.POSITION)
        print(get_robot_data(robot))
        time.sleep(0.005)

    # Move to standing position
    standing_pose = pos_transform(np.zeros(12))
    for t in tqdm(range(300)):
        blend_ratio = np.minimum(t / 200., 1)
        action = (1 - blend_ratio) * current_motor_angle + blend_ratio * standing_pose
        robot.Step(action, robot_config.MotorControlMode.POSITION)
        print(get_robot_data(robot))
        time.sleep(0.005)

    model = checkpoint.model
    model.eval()

    control_timestep = 0.03
    physics_timestep = 0.001
    start_time = time.time()
    last_action = standing_pose
    last_action_time = 0
    
    try:
        while time.time() - start_time < 600:
            current_time = time.time()
            if current_time - last_action_time >= control_timestep:
                current_motor_angles = np.array(robot._motor_angles)
                joint_torques = -np.array(robot._observed_motor_torques)
                joint_velocities = np.array(robot.GetMotorVelocities())
                foot_contacts = np.array(robot._raw_state.footForce)
                obs = {
                    'a1/joints_pos': current_motor_angles.reshape(1, -1),
                    'a1/joints_trq': joint_torques.reshape(1, -1),
                    'a1/joints_vel': joint_velocities.reshape(1, -1),
                    'a1/sensors_foot': foot_contacts.reshape(1, -1)
                }
                action = predict_action(obs, model)
                action = legs2_to_legs3(action)
                desired_motor_angles = pos_transform(action)
                robot.Step(desired_motor_angles, robot_config.MotorControlMode.POSITION)

                print_colored("Time:", current_time - start_time, 'red')
                print_colored("Observs:", obs, 'magenta')
                print_colored("Torques:", joint_torques, 'yellow')
                print_colored("Actions:", action, 'green')
                print_colored("Command:", desired_motor_angles, 'blue')

                last_action_time = current_time
                last_action = desired_motor_angles

            robot.Step(last_action, robot_config.MotorControlMode.POSITION)
            time.sleep(physics_timestep)

    except Exception as e:
        print_colored('Error: ', str(e), '91')
    finally:
        robot.Terminate()

if __name__ == '__main__':
    app.run(main)
