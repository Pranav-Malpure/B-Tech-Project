# The basic mujoco wrapper.
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

import dm_env
from dm_env import Environment
from dm_control.composer.environment import Environment # Even though this may be labelled as blue in VSCode, it is a class from the file 
  
from forward_kinematics import forward_kinematics, get_angles
from dm_control import composer as _composer
from agent import SACAgent

from utils.hyperparameters import hyperparams
from utils.plot_utils import plot_rewards
import moviepy.editor as mp

duration = 250    # (seconds)
framerate = 30  # (Hz)

# import torch
# device = torch.device('mps')

import mujoco.viewer as gui_viewer
from Task_reach import Reach_task, reach_site_vision, reach_site_features

# Function used to save simulated videos
def save_video(frames, output_path, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 300
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0], interpolation='nearest')

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                    interval=interval, blit=True, repeat=False)

    anim.save(output_path, writer='ffmpeg', fps=framerate, codec='mpeg4', dpi=dpi, bitrate=8000)
    plt.close(fig)

    # Example usage
    frames = [...]  # List of frames
    output_path = 'animation.gif'
    # save_video(frames, output_path, framerate=30)

class SACModel(nn.Module):
    def __init__(self, obs, action_size):
        super().__init__()
        self.action_size = action_size-3
        obs_size = obs
        print("SAC class", obs_size)
        print("SAC class action", self.action_size)

        self.actor = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size*2),
            Lambda(self.squashed_diagonal_gaussian_head)
        )
        torch.nn.init.xavier_uniform_(self.actor[0].weight)
        torch.nn.init.xavier_uniform_(self.actor[2].weight)
        torch.nn.init.xavier_uniform_(self.actor[4].weight, gain=1)
        self.q_func1, self.q_func1_optimizer = self.make_q_func_with_optimizer(obs_size, self.action_size)
        self.q_func2, self.q_func2_optimizer = self.make_q_func_with_optimizer(obs_size, self.action_size)

    def make_q_func_with_optimizer(self, obs_size, action_size):
        q_func = nn.Sequential(
            pfrl.nn.ConcatObsAndAction(),
            nn.Linear(obs_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        torch.nn.init.xavier_uniform_(q_func[1].weight)
        torch.nn.init.xavier_uniform_(q_func[3].weight)
        torch.nn.init.xavier_uniform_(q_func[5].weight)
        q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=3e-4)
        return q_func, q_func_optimizer
    def squashed_diagonal_gaussian_head(self, x):
        assert x.shape[-1] == self.action_size * 2
        mean, log_scale = torch.chunk(x, 2, dim=1)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # cache_size=1 is required for numerical stability
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
        )

np.random.seed(10)

frames=[]

_ReachWorkspace = collections.namedtuple(
    '_ReachWorkspace', ['target_bbox', 'tcp_bbox', 'arm_offset'])
_PROP_Z_OFFSET = 0.001
_DUPLO_WORKSPACE = _ReachWorkspace(
    target_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, _PROP_Z_OFFSET),
        upper=(0.1, 0.1, _PROP_Z_OFFSET)),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, 0.2),
        upper=(0.1, 0.1, 0.4)),
    arm_offset=robots.ARM_OFFSET)

_SITE_WORKSPACE = _ReachWorkspace(
    target_bbox=workspaces.BoundingBox(
        lower=(-0.2, -0.2, 0.02),
        upper=(0.2, 0.2, 0.4)),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-0.2, -0.2, 0.02),
        upper=(0.2, 0.2, 0.4)),
    arm_offset=robots.ARM_OFFSET)

task_object = reach_site_features()
# task_object = reach_site_vision()

env = _composer.Environment(task = task_object, time_limit = duration, random_state=3)
action_dim = env.action_spec().shape[0]
action_spec = env.action_spec()
state_dim = np.size(env.random_state.uniform(
      low=action_spec.minimum, high=action_spec.maximum,).astype(action_spec.dtype, copy=False))

model = SACModel(6, action_dim)

optimizer = optim.Adam(model.parameters(), lr = 3e-4)
policy = model.actor

torch.nn.init.xavier_uniform_(policy[0].weight)
torch.nn.init.xavier_uniform_(policy[2].weight)
torch.nn.init.xavier_uniform_(policy[4].weight, gain=1)
policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

agent = SoftActorCritic(
    policy=model.actor,
    q_func1=model.q_func1,
    q_func2=model.q_func2,
    policy_optimizer=policy_optimizer,
    q_func1_optimizer=model.q_func1_optimizer,
    q_func2_optimizer=model.q_func2_optimizer,
    replay_buffer=replay_buffers.ReplayBuffer(capacity=10 ** 6),
    gamma=0.99,
    # phi = lambda x: x.astype('float32', copy=False), 
    gpu=-1, #for using mps, set to 0. For CUDA, set to 1
)

reward_plot = []
episode_plot = []

for episode in range(100):
    print()
    print("New Episode starts")
    obs = env.reset()
    obs_ = obs[3] # This is to provide the first observation on line 231
    R = 0 
    t = 0  
    counter = 0
    max_episode_len = float('inf')
    while True:
        obs_pos = torch.tensor(get_angles(obs_['jaco_arm/joints_pos'][0]), dtype=torch.float32)
        action = agent.act(obs_pos)
        extend = np.array([0,0,0])
        action = np.concatenate((action, extend))
        time_step = env.step(action)
        
        reward = time_step.reward
        if reward == None:
            continue
        
        if time_step.step_type == dm_env.StepType.FIRST:
            done = False
        elif time_step.step_type == dm_env.StepType.MID:
            done = False
        elif time_step.step_type == dm_env.StepType.LAST:
            print("Hello there:)")
            done = True

        obs_ = time_step[3]
        obs_pos = torch.tensor(get_angles(obs_['jaco_arm/joints_pos'][0]), dtype=torch.float32)
        agent.observe(obs_pos, reward, done, reset=False)
        R += reward
        t += 1

        if done:
            print("episode done")
            break

        counter += 1

    if episode % 1 == 0:
        print(f'Episode: {episode + 1}, Total Reward: {R}')
        print("counter", counter)
        reward_plot.append(R)
        episode_plot.append(episode+1)
    if episode % 10 == 0:
        print('statistics:', agent.get_statistics())


plt.plot(episode_plot, reward_plot, '-')
plt.xlabel("Episode Number")
plt.ylabel("Total Reward")
plt.title("Reward v/s Episode Number")
plt.show()
plt.savefig('reward_plot_same_initial.png')
agent.save('test_agent')
print("TRAINING DONE")

exit() # Below this is to evaluate the trained models

agent.load('test_agent')

visual_env = _composer.Environment(reach_site_vision(), time_limit=duration)

frames = []
eval_rewards = []
eval_episodes = []
with agent.eval_mode():
    for i in range(10):
        obs = env.reset()
        # obs =  env.step([0, 0, 0, 0, 0, 0, 1, 1, 1])
        obs_ = obs[3]
        visual_env.reset()
        R = 0
        t = 0
        while True:
            # obs_pos = (get_angles(obs_['jaco_arm/joints_pos'][0]))
            obs_pos = obs_['jaco_arm/joints_torque'][0]
            # obs_pos = get_angles(obs_['jaco_arm/joints_pos'][0])
            action = agent.act(torch.tensor(obs_pos, dtype=torch.float32))
            extend = np.array([0,0,0])
            action = np.concatenate((action, extend))   
            time_step = env.step(action)
            r = time_step.reward
            if r == None:
                continue
            if time_step.step_type == 1:
                done = False
            elif time_step.step_type == 2:
                done = True
            obs_ = time_step[3]
            # obs_pos = torch.tensor(get_angles(obs_['jaco_arm/joints_pos'][0]), dtype=torch.float32)
            obs_pos = torch.tensor(obs_['jaco_arm/joints_torque'][0], dtype=torch.float32)
            # obs, r, done, _ = env.step(action)
            R += r
            t += 1
            reset = t > 200
            agent.observe(obs_pos, r, done, reset)
            timestep_visual = visual_env.step(action)
            frames.append(timestep_visual.observation['front_close'])
            if done or reset:
                break
        print('evaluation episode:', i, 'R:', R)
        eval_rewards.append(R)
        eval_episodes.append(i+1)

        all_frames = np.concatenate(frames, axis=0)
        filename = f'reach_vision_testing_hd_mpeg4_{i}.gif'
        save_video(all_frames, filename, 30)
        frames = []

for i in range(10):
    filename = f'reach_vision_testing_hd_mpeg4_{i}.gif'
    filename_mp4 = f'{i}.mp4'
    clip = mp.VideoFileClip(filename)
    clip.write_videofile(filename_mp4)
torch.save(model.state_dict(), 'sac_model.pth')

exit()

