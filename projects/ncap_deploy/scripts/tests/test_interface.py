# Does not execute anything on the robot

from robot_interface import RobotInterface # pytype: disable=import-error

i = RobotInterface()
o = i.receive_observation()