from dm_control import mujoco
import gymnasium as gym
from gym import wrappers
import pfrl
from pfrl import replay_buffers, experiments
from pfrl.agents import SoftActorCritic
from pfrl.nn.lmbda import Lambda

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions
# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib
import numpy as np
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC
#@title Imports
# General
import copy
import os
import itertools
# from IPython.display import clear_output
import numpy as np
import collections

import functools
# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
from IPython import display
import PIL.Image
# The basic mujoco wrapper.
from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation
# from dm_control.rl.control import Environment
from dm_control.manipulation.shared import workspaces
from dm_control.manipulation.shared import robots
from dm_control import manipulation
import dm_control.suite as suite
from dm_control.manipulation.shared import observations
from dm_control.manipulation.shared import arenas

# from dm_control.locomotion import

from dm_control.entities.manipulators import kinova
from dm_control import viewer
from dm_control.composer import Task
from dm_control.composer.initializers import ToolCenterPointInitializer
from dm_env import Environment
from dm_control.composer.environment import Environment # Even though this may be labelled as blue, it is a class from the file 
  
from forward_kinematics import forward_kinematics, get_angles
from dm_control import composer as _composer
from agent import SACAgent

from utils.hyperparameters import hyperparams
from utils.plot_utils import plot_rewards
import moviepy.editor as mp

# from dmc

duration = 500    # (seconds)
framerate = 30  # (Hz)

import mujoco.viewer as gui_viewer
from Task_reach import Reach_task, reach_site_vision, reach_site_features



name = 'jaco_arm'
arm = kinova.JacoArm(name)
arm._build()
arm_observables = arm._build_observables()
# print(arm)
hand = kinova.JacoHand()
arm.attach(hand)


a = [0,0,0]
