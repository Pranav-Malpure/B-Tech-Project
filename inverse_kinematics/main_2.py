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
        
        # self.obs_shape = obs
        self.action_size = action_size-3
        obs_size = obs.shape[1]
        print("SAC class", obs_size)
        print("SAC class action", self.action_size)
        self.actor = nn.Sequential(
            nn.Conv2d(obs_size, 32, kernel_size=4, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
            nn.Tanh()
        )
        # input_size = torch.prod(torch.tensor(joints_pos_size)).item()
        # print("input_size", input_size)
        # self.actor = nn.Sequential(
        #     nn.Linear(obs_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.action_size*2),
        #     Lambda(self.squashed_diagonal_gaussian_head)
        # )
        torch.nn.init.xavier_uniform_(self.actor[0].weight)
        torch.nn.init.xavier_uniform_(self.actor[2].weight)
        torch.nn.init.xavier_uniform_(self.actor[4].weight, gain=1)
        torch.nn.init.xavier_uniform_(self.actor[7].weight, gain=1)
        torch.nn.init.xavier_uniform_(self.actor[9].weight, gain=1)

        # self.q1 = pfrl.q_functions.FCQuadraticStateQFunction(
        #     n_input_channels=joints_pos_size,
        #     n_dim_action=action_size,
        #     n_hidden_channels=256,
        #     n_hidden_layers=3,
        #     action_space=action_size
        # )

        # self.q2 = pfrl.q_functions.FCQuadraticStateQFunction(
        #     # pfrl.nn.ConcatObsAndAction(),
        #     n_input_channels=joints_pos_size,
        #     n_dim_action=action_size,
        #     n_hidden_channels=256,
        #     n_hidden_layers=3,
        #     action_space=action_size
        # )
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
        # print("input from linear", x.shape)
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

np.random.seed(4)


# Visualize the joint axis
scene_option = mujoco.wrapper.core.MjvOption()
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True


name = 'jaco_arm'
arm = kinova.JacoArm(name)
arm._build()
arm_observables = arm._build_observables()
# print(arm)
hand = kinova.JacoHand()
# arm.attach(hand)
mjcf_physics_instance = mjcf.Physics.from_xml_path("/Users/pranavmalpure/B-Tech-Project/btp/lib/python3.9/site-packages/dm_control/third_party/kinova/jaco_arm.xml")
mjcf_physics_instance = mjcf.Physics.from_mjcf_model(arm._mjcf_root)
"""print(arm.observables._observables['joints_pos']._raw_callable(mjcf_physics_instance))  # get position of joints"""
# print(arm.observables._observables['joints_vel']._callable(mjcf_physics_instance)())
# print(arm.observables._observables['joints_pos'])

"""
print(arm_observables.joints_pos._raw_callable(mjcf_physics_instance)) # both this and the above are alternate methods of getting the same thing
end_effector_pose = forward_kinematics(arm_observables.joints_pos._raw_callable(mjcf_physics_instance)) 
print("End effector pose is %s"%end_effector_pose[0])
print("End effector pose is %s"%end_effector_pose[1])
"""
# print(arm.)
# print(arm.observables)

frames=[]
# while mjcf_physics_instance.data.time < duration/2:
#     arm.configure_joints(mjcf_physics_instance, [1,1,1,1,1,1])
#     mjcf_physics_instance.step()
#     if len(frames) < mjcf_physics_instance.data.time * framerate:
#         pixels = mjcf_physics_instance.render(scene_option=scene_option)
#         frames.append(pixels)

# Each time step is 0.02 seconds
mjcf_physics_instance_kinova = arm._mjcf_root.get_assets()
mjcf_physics_instance_kinova = mujoco.Physics.from_xml_string(xml_string=arm._mjcf_root.to_xml_string(), assets= arm._mjcf_root.get_assets())

# print(type(mjcf_physics_instance_kinova))
# print(type(mjcf_physics_instance))
# mjcf_physics_instance_kinova.model.set_control([0,0,0,0,0,0])


arm.configure_joints(mjcf_physics_instance, [0, 0, 0, 0, 0, 0])
"""
print(arm_observables.joints_pos._raw_callable(mjcf_physics_instance)) # both this and the above are alternate methods of getting the same thing
print("End effector pose now is %s"%forward_kinematics(arm_observables.joints_pos._raw_callable(mjcf_physics_instance))[0])
print("configuring now")
arm.configure_joints(mjcf_physics_instance, [1, 1, 1, 1, 1, 1])
print("End effector pose now is %s"%forward_kinematics(arm_observables.joints_pos._raw_callable(mjcf_physics_instance))[0])

mjcf_physics_instance.step()
print(arm_observables.joints_pos._raw_callable(mjcf_physics_instance)) # both this and the above are alternate methods of getting the same thing
mjcf_physics_instance.step()
"""
# gui_viewer.launch_from_path(mjcf_physics_instance)


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

arena = arenas.Standard()
# task_object = Reach_task(arena=arena, arm=arm, hand=hand, prop=None, obs_settings=observations.PERFECT_FEATURES, workspace=_SITE_WORKSPACE, control_timestep=0.02)
# task_object = reach_site_features()
task_object = reach_site_vision()

env = _composer.Environment(task = task_object, time_limit = duration, random_state=1)
action_dim = env.action_spec().shape[0]
action_spec = env.action_spec()
state_dim = np.size(env.random_state.uniform(
      low=action_spec.minimum, high=action_spec.maximum,).astype(action_spec.dtype, copy=False))
# print("action_spec, ", action_dim)
# print("state_spec, ", state_dim)

obs = env.step([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1])

print(obs[3])
print(obs[3]['jaco_arm/joints_pos'])
print(get_angles(obs[3]['jaco_arm/joints_pos'][0]))

# obs_shape = np.shape(np.squeeze(obs[3]['front_close']))
# obs_shape = np.shape(obs[3]['front_close'])
# print(np.shape(obs[3]['front_close']))
# print(obs_shape)
# reward = env.task.get_reward()
joints_pos = torch.tensor(obs[3]['front_close'][0], dtype=torch.float32)
# joints_pos = torch.FloatTensor(obs[3]['jaco_arm/joints_pos'])
print("joints_pos = ", joints_pos)
print("joints_pos_type = ", type(joints_pos))

print("joints_pos.shape = ", joints_pos.shape)
print("joints_pos.shape[1] = ", joints_pos.shape[1])
print("action_dim = ", action_dim)
model = SACModel(joints_pos, action_dim)


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
    gpu=-1,
)

experiment = copy.deepcopy(torch.tensor(obs[3]['front_close']))
# experiments.train_agent_batch_with_evaluation(
#     agent = agent, env=make_batch_env(test=False),
# )
# print("SHAPE?? ",(torch.tensor(obs[3]['jaco_arm/joints_pos'])).flatten(start_dim=1).shape)
# print("SHAPE?? ",(torch.tensor(obs[3]['jaco_arm/joints_pos'])).flatten(start_dim=1).flatten().shape)
# print(agent.policy(torch.tensor(obs[3]['jaco_arm/joints_pos']).flatten(start_dim=1)))
# exit()
# agent.load('trained_agent')


reward_plot = []
episode_plot = []

for episode in range(10):
    obs = env.reset()
    # obs = env.step([0, 0, 0, 0, 0, 0, 1, 1, 1])
    obs_ = obs[3]
    # print(obs_)
    # obs_ =  np.array(get_angles(obs_['jaco_arm/joints_pos'][0]))
    # # print("size of obs from step", obs_.shape)
    # action = agent.act(obs_)
    # print("actionnnnnn", action)
    # agent.observe(obs_, 0.2, False, False)
    # print("executed observe")
    R = 0 
    t = 0  
    counter = 0
    max_episode_len = float('inf')
    while True:
        # print("While loop counter: ",counter)

        obs_pos = torch.tensor(obs_['front_close'][0], dtype=torch.float32)
        # obs_pos = torch.from_numpy(np.array(get_angles(obs_['jaco_arm/joints_pos'][0])))
        # obs_pos = torch.tensor(obs_pos, dtype=torch.float32)
        # obs_pos = get_angles(obs_['jaco_arm/joints_pos'][0])
        # obs_pos = np.array(get_angles(obs_['jaco_arm/joints_pos'][0]))
        # print("obs_pos inside the while loop", obs_pos.shape)
                           
        # obs_pos = torch.FloatTensor(obs_['jaco_arm/joints_pos'])
        # obs_image = obs_image.squeeze(0).permute(2,1,0)
        # obs_image.permute(2,0,1)
        # action = agent.act(obs_image)
        # action = agent.act(obs_image.permute(0,3,1,2))
        # print("obs_shape ", obs_pos.shape)
        action = agent.act(obs_pos)
        extend = np.array([0,0,0])
        action = np.concatenate((action, extend))
        time_step = env.step(action)
        reward = time_step.reward
        if reward == None:
            continue
        if time_step.step_type == 1:
            done = False
        elif time_step.step_type == 2:
            done = True
        
        # next_obs = time_step.observation['front_close']
        obs_ = time_step[3]
        # reset = t > max_episode_len
        # reset = False 
        obs_pos = torch.tensor(obs_['front_close'][0], dtype=torch.float32)
        # obs_pos = torch.from_numpy(np.array(get_angles(obs_['jaco_arm/joints_pos'][0])))
        # obs_pos = torch.tensor(obs_pos, dtype=torch.float32)
        agent.observe(obs_pos, reward, done, reset=False)

        # experiences = [[{'state': obs}],[{'action': action}],[{'reward': reward}],[{'is_state_terminal': done}],[{'next_state': next_obs}]]
        # agent.update(experiences)

        # obs_ = next_obs
        R += reward
        t += 1

        if done:
            break
        counter += 1

    if episode % 1 == 0:
        print(f'Episode: {episode + 1}, Total Reward: {R}')
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

exit()
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
            obs_pos = (get_angles(obs_['jaco_arm/joints_pos'][0]))
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
            obs_pos = torch.tensor(get_angles(obs_['jaco_arm/joints_pos'][0]), dtype=torch.float32)
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
# plt.plot(eval_episodes, eval_rewards, '-')
# plt.xlabel("Episode Number")
# plt.ylabel("Total Reward")
# plt.title("Evaluation of trained agent-1")
# plt.savefig('eval_untrained.png')
for i in range(10):
    filename = f'reach_vision_testing_hd_mpeg4_{i}.gif'
    filename_mp4 = f'{i}.mp4'
    clip = mp.VideoFileClip(filename)
    clip.write_videofile(filename_mp4)
torch.save(model.state_dict(), 'sac_model.pth')

exit()


model = SAC(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)
model.save("sac_arm")
del model

model = SAC.load("sac_arm")
obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    viewer.launch(env)

exit("Before this is stable baselines implementation")
agent = SACAgent(state_dim, action_dim, **hyperparams)

rewards = []
num_episodes = 10
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        
        if done:
            break
    rewards.append(episode_reward)

plot_rewards(rewards, save_path="images_gif/training_results.png")
exit()
print(type(env.action_spec()), "Helllllloooo")
print((env.action_spec()))


print("HELLO")

print(env._task.arm.observables._observables['joints_pos']._raw_callable(env.physics)) #prints the joints states in the form of sin, cos theta
env._task.arm.configure_joints(env.physics, [0.3,0.5,0.6,0.7,1,0])
print()
print(env._task.arm.observables._observables['joints_pos']._raw_callable(env.physics)) #prints the joints states in the form of sin, cos theta
frames = []
timestep = env.reset()

count = 0
while not timestep.last():
    timestep = env.step([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1])
    frames.append(timestep.observation['front_close'])
    count += 1
all_frames = np.concatenate(frames, axis=0)
print("count", count)
save_video(all_frames, 'reach_vision_testing_hd_mpeg4.gif', 30)
print(env._task.arm.observables._observables['joints_pos']._raw_callable(env.physics)) #prints the joints states in the form of sin, cos theta


