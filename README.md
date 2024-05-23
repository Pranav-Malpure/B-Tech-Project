# RL based control of a 6-DOF robotic arm for inverse kinematics

Used `dm_control` to simulate and implement a RL environment for Kinova Jaco Arm, and applied the Soft-Actor Critic method from `pfrl` library for training the arm to move to any point in 3D space from any random position.

The `main.py` file inside the `inverse_kinematics` directory contains the main code for running the environment, and the file `Task_reach.py` file contains the reward and observation structure.

Find the report [here](https://github.com/Pranav-Malpure/B-Tech-Project/blob/main/report.pdf) for more details about the implementation and the results.

_This project was my B-Tech Project in the Aerospace Department at IIT Bombay._
