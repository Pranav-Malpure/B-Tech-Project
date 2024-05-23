# @author = Pranav Malpure - https://github.com/Pranav-Malpure

# The basic mujoco wrapper.
import dm_control
import torch
from dm_control import mujoco

import pfrl
from pfrl import replay_buffers, experiments
from pfrl.agents import SoftActorCritic
from pfrl.nn.lmbda import Lambda

import torch.nn as nn
import torch.optim as optim
from torch import distributions
# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib
import numpy as np
import math

#@title Imports
# General
import copy
import os
import itertools
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
from dm_control.composer.environment import Environment # Even though this may be colour coded as blue in VScode, it is a class from the file, not a function
  
from forward_kinematics import forward_kinematics, get_angles
from dm_control import composer as _composer
from agent import SACAgent

from utils.hyperparameters import hyperparams
from utils.plot_utils import plot_rewards
import moviepy.editor as mp
import pickle

duration = 15   # (seconds)
framerate = 30  # (Hz)

# import torch
# device = torch.device('mps')

import mujoco.viewer as gui_viewer
from Task_reach import Reach_task, reach_site_vision, reach_site_features

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

class SACModel(nn.Module):
    def __init__(self, obs, action_size):
        super().__init__()
        self.action_size = action_size-3
        obs_size = obs
        self.actor = nn.Sequential(
            nn.Linear(obs_size, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, self.action_size*2),
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
            nn.Linear(obs_size + action_size, 15),
            nn.ReLU(),
            nn.Linear(15, 9),
            nn.ReLU(),
            nn.Linear(9, 1),
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



np.random.seed(859110)

# Visualize the joint axis
scene_option = mujoco.wrapper.core.MjvOption()
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True


name = 'jaco_arm'
arm = kinova.JacoArm(name)
arm._build(name)
arm_observables = arm._build_observables()
hand = kinova.JacoHand()

task_object = reach_site_features()

env = _composer.Environment(task = task_object, time_limit = duration, random_state=3)
action_dim = env.action_spec().shape[0]
action_spec = env.action_spec()
state_dim = np.size(env.random_state.uniform(
      low=action_spec.minimum, high=action_spec.maximum,).astype(action_spec.dtype, copy=False))

model = SACModel(15, action_dim)

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
    gpu=-1, #for using mps, set to 0, else -1
    # initial_temperature=3,
    # entropy_target=0,
    # temperature_optimizer_lr = 0.00005
)

# Uncomment the below for continuing the training and loading the previous variables

# agent.load('end_effector_pos_model_less_neurons')
# with open('end_effector_pos_model_less_neurons_plot/reward_plot', 'rb') as file:
#     reward_plot = pickle.load(file)
# with open('end_effector_pos_model_less_neurons_plot/episode_plot', 'rb') as file:
#     episode_plot = pickle.load(file)
# with open('end_effector_pos_model_less_neurons_plot/episode_length_plot', 'rb') as file:
#     episode_length_plot = pickle.load(file)
# with open('end_effector_pos_model_less_neurons_plot/average_q1_list', 'rb') as file:
#     average_q1_list = pickle.load(file)
# with open('end_effector_pos_model_less_neurons_plot/average_q2_list', 'rb') as file:
#     average_q2_list = pickle.load(file)
# with open('end_effector_pos_model_less_neurons_plot/average_q_func1_loss_list', 'rb') as file:
#     average_q_func1_loss_list = pickle.load(file)
# with open('end_effector_pos_model_less_neurons_plot/average_q_func2_loss_list', 'rb') as file:
#     average_q_func2_loss_list = pickle.load(file)
# with open('end_effector_pos_model_less_neurons_plot/n_updates_list', 'rb') as file:
#     n_updates_list = pickle.load(file)
# with open('end_effector_pos_model_less_neurons_plot/average_entropy_list', 'rb') as file:
#     average_entropy_list = pickle.load(file)
# with open('end_effector_pos_model_less_neurons_plot/temperature_list', 'rb') as file:
#     temperature_list = pickle.load(file)
# # with open('end_effector_pos_model_less_neurons_plot/counter_list', 'rb') as file:
# #     counter_list = pickle.load(file)
# with open('end_effector_pos_model_less_neurons_plot/episode_done', 'rb') as file:
#     episode_done = pickle.load(file)

# Comment the below 12 lines for continuing training from a saved instance and saved variables
reward_plot = []
episode_plot = []
episode_length_plot = []
average_q1_list = []
average_q2_list = []
average_q_func1_loss_list = []
average_q_func2_loss_list = []
n_updates_list = []
average_entropy_list = []
temperature_list = []
counter_list = []
episode_done = []

prev_reward = 0
try:
    for episode in range(5000):
        print()
        print("New Episode:", (episode+1),"starts" )
        obs = env.reset()
        obs_ = obs[3]
        R = 0 
        t = 0  
        counter = 0
        max_episode_len = float('inf')
        while True: 
            # print("While loop counter: ",counter)
            obs_pos = torch.tensor(get_angles(obs_['jaco_arm/joints_pos'][0]), dtype=torch.float32)
            obs_vel = torch.tensor(obs_['jaco_arm/joints_vel'][0], dtype=torch.float32)
            obs_cartesian_pos = torch.tensor(obs_['jaco_arm/jaco_hand/pinch_site_pos'][0], dtype=torch.float32)
            obs_tensor = torch.cat((obs_pos,obs_vel, obs_cartesian_pos), dim=0)
          
            action = agent.act(obs_tensor)
            extend = np.array([0,0,0])
            action = np.concatenate((action, extend))
            action = (action + 1)*math.pi
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
            obs_vel = torch.tensor(obs_['jaco_arm/joints_vel'][0], dtype=torch.float32)
            obs_cartesian_pos = torch.tensor(obs_['jaco_arm/jaco_hand/pinch_site_pos'][0], dtype=torch.float32)
            obs_tensor = torch.cat((obs_pos, obs_vel, obs_cartesian_pos), dim=0)

            agent.observe(obs_tensor, reward, done, reset=False)
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
        episode_statistics = agent.get_statistics()
        average_q1_list.append(episode_statistics[0][1])
        average_q2_list.append(episode_statistics[1][1])
        average_q_func1_loss_list.append(episode_statistics[2][1])
        average_q_func2_loss_list.append(episode_statistics[3][1])
        n_updates_list.append(episode_statistics[4][1])
        average_entropy_list.append(episode_statistics[5][1])
        temperature_list.append(episode_statistics[6][1])
        episode_length_plot.append(counter)
        if counter < 374:
            episode_done.append(1)
            agent.initial_temperature=1
        else:
            episode_done.append(0)
        
    agent.save('end_effector_pos_model_less_neurons')
    with open('end_effector_pos_model_less_neurons_plot/reward_plot', 'wb') as file:
        pickle.dump(reward_plot, file)
    with open('end_effector_pos_model_less_neurons_plot/episode_plot', 'wb') as file:
        pickle.dump(episode_plot, file)
    with open('end_effector_pos_model_less_neurons_plot/episode_length_plot', 'wb') as file:
        pickle.dump(episode_length_plot, file)
    with open('end_effector_pos_model_less_neurons_plot/average_q1_list', 'wb') as file:
        pickle.dump(average_q1_list, file)
    with open('end_effector_pos_model_less_neurons_plot/average_q2_list', 'wb') as file:
        pickle.dump(average_q2_list, file)
    with open('end_effector_pos_model_less_neurons_plot/average_q_func1_loss_list', 'wb') as file:
        pickle.dump(average_q_func1_loss_list, file)
    with open('end_effector_pos_model_less_neurons_plot/average_q_func2_loss_list', 'wb') as file:
        pickle.dump(average_q_func2_loss_list, file)
    with open('end_effector_pos_model_less_neurons_plot/n_updates_list', 'wb') as file:
        pickle.dump(n_updates_list, file)
    with open('end_effector_pos_model_less_neurons_plot/average_entropy_list', 'wb') as file:
        pickle.dump(average_entropy_list, file)
    with open('end_effector_pos_model_less_neurons_plot/temperature_list', 'wb') as file:
        pickle.dump(temperature_list, file)
    with open('end_effector_pos_model_less_neurons_plot/episode_done', 'wb') as file:
        pickle.dump(episode_done, file)
    plt.plot(episode_plot, reward_plot, '-', linewidth = 0.7)
    plt.xlabel("Episode Number")
    plt.ylabel("Total Reward")
    plt.title("Reward v/s Episode Number")
    plt.show()
    plt.savefig('reward_plot_same_initial.png')
    print("TRAINING DONE")
except Exception as e:
    # pass
    agent.save('end_effector_pos_model_less_neurons')
    with open('`end_effector_pos_model_less_neurons_plot`/reward_plot', 'wb') as file:
        pickle.dump(reward_plot, file)
    with open('end_effector_pos_model_less_neurons_plot/episode_plot', 'wb') as file:
        pickle.dump(episode_plot, file)
    with open('end_effector_pos_model_less_neurons_plot/episode_length_plot', 'wb') as file:
        pickle.dump(episode_length_plot, file)
    with open('end_effector_pos_model_less_neurons_plot/average_q1_list', 'wb') as file:
        pickle.dump(average_q1_list, file)
    with open('end_effector_pos_model_less_neurons_plot/average_q2_list', 'wb') as file:
        pickle.dump(average_q2_list, file)
    with open('end_effector_pos_model_less_neurons_plot/average_q_func1_loss_list', 'wb') as file:
        pickle.dump(average_q_func1_loss_list, file)
    with open('end_effector_pos_model_less_neurons_plot/average_q_func2_loss_list', 'wb') as file:
        pickle.dump(average_q_func2_loss_list, file)
    with open('end_effector_pos_model_less_neurons_plot/n_updates_list', 'wb') as file:
        pickle.dump(n_updates_list, file)
    with open('end_effector_pos_model_less_neurons_plot/average_entropy_list', 'wb') as file:
        pickle.dump(average_entropy_list, file)
    with open('end_effector_pos_model_less_neurons_plot/temperature_list', 'wb') as file:
        pickle.dump(temperature_list, file)
    with open('end_effector_pos_model_less_neurons_plot/episode_done', 'wb') as file:
        pickle.dump(episode_done, file)
    raise e


agent.load('end_effector_pos_model_less_neurons')

agent.change_initial_temperature(0)

frames = []
eval_rewards = []
eval_episodes = []
# with agent.eval_mode(), viewer.launch(env, agent.policy) as env:
for i in range(20):
    obs = env.reset()
    # obs =  env.step([0, 0, 0, 0, 0, 0, 1, 1, 1])
    obs_ = obs[3]
    # visual_env.reset()
    R = 0
    t = 0
    counter = 0
    while True:
        counter +=1
        obs_pos = (get_angles(obs_['jaco_arm/joints_pos'][0]))
        # obs_pos = obs_['jaco_arm/joints_torque'][0]
        obs_pos = torch.tensor(get_angles(obs_['jaco_arm/joints_pos'][0]), dtype=torch.float32)
        obs_vel = torch.tensor(obs_['jaco_arm/joints_vel'][0], dtype=torch.float32)
        obs_cartesian_pos = torch.tensor(obs_['jaco_arm/jaco_hand/pinch_site_pos'][0], dtype=torch.float32)
        obs_tensor = torch.cat((obs_pos, obs_vel, obs_cartesian_pos), dim=0)
        # obs_pos = get_angles(obs_['jaco_arm/joints_pos'][0])
        action = agent.act(torch.tensor(obs_tensor, dtype=torch.float32))
        extend = np.array([0,0,0])
        action = np.concatenate((action, extend))  
        action = (action+1)*np.pi 
        time_step = env.step(action)
        # print("ACTION TAKEN: ", action)
        r = time_step.reward
        if r == None:
            continue
        if time_step.step_type == 1:
            done = False
        elif time_step.step_type == 2:
            done = True
        obs_ = time_step[3]
        obs_pos = torch.tensor(get_angles(obs_['jaco_arm/joints_pos'][0]), dtype=torch.float32)
        obs_vel = torch.tensor(obs_['jaco_arm/joints_vel'][0], dtype=torch.float32)
        obs_cartesian_pos = torch.tensor(obs_['jaco_arm/jaco_hand/pinch_site_pos'][0], dtype=torch.float32)
        obs_tensor = torch.cat((obs_pos, obs_vel, obs_cartesian_pos), dim=0)
        R += r
        t += 1
        frames.append(env.physics.render(height = 480, width = 640))
        if done:
            break
    print('evaluation episode:', i, 'R:', R)
    eval_rewards.append(R)
    eval_episodes.append(i+1)
    all_frames = frames
    filename = f'YOOreach_vision_testing_hd_mpeg4_{i}.gif'
    save_video(all_frames, filename, 30)
    frames = []
    print()
# plt.plot(eval_episodes, eval_rewards, '-')
# plt.xlabel("Episode Number")
# plt.ylabel("Total Reward")
# plt.title("Evaluation of trained agent-1")
# plt.savefig('eval_untrained.png')

# for i in range(10):
#     filename = f'reach_vision_testing_hd_mpeg4_{i}.gif'
#     filename_mp4 = f'{i}.mp4'
#     clip = mp.VideoFileClip(filename)
#     clip.write_videofile(filename_mp4)
# torch.save(model.state_dict(), 'sac_model.pth')


