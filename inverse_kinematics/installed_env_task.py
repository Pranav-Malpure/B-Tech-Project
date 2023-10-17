# The basic mujoco wrapper.
from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

import numpy as np

#@title Imports
# General
import copy
import os
import itertools
# from IPython.display import clear_output
import numpy as np

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
from dm_control.rl.control import Environment
from dm_control import manipulation
import dm_control.suite as suite
# from dm_control.locomotion import

from dm_control.entities.manipulators import kinova
from dm_control import viewer
from dm_control.composer import Task
  
duration = 20    # (seconds)
framerate = 30  # (Hz)

import tensorflow as tf
#incomoplete task
class inverse_kinematics_task(Task):
    def __init__(self):
        pass
    
    def root_entity(self):
        pass


def save_video(frames, output_path, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                    interval=interval, blit=True, repeat=False)

    anim.save(output_path, writer='ffmpeg', fps=framerate)
    plt.close(fig)

    # Example usage
    frames = [...]  # List of frames
    output_path = 'animation.gif'
    # save_video(frames, output_path, framerate=30)
np.random.seed(42)

def calculate_distance(position1, position2):
    return np.sqrt(np.sum((np.array(position1) - np.array(position2))**2))


env = manipulation.load('reach_site_features', seed=42)
action_spec = env.action_spec()
print(type(env.action_spec()), "Helllllloooo")
print((env.action_spec()))
print("HELLOOO ", env.random_state.uniform(
      low=action_spec.minimum,
      high=action_spec.maximum,
  ).astype(action_spec.dtype, copy=False))
print(np.shape(env.random_state.uniform(
      low=action_spec.minimum,
      high=action_spec.maximum,
  ).astype(action_spec.dtype, copy=False)))

def sample_random_action():
  return env.random_state.uniform(
      low=action_spec.minimum,
      high=action_spec.maximum,
  ).astype(action_spec.dtype, copy=False)

frames = []

timestep = env.step([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1])
# frames.append(timestep.observation['front_close'])
print(timestep)
print(timestep[3][])
env

exit()

print("HELLO")

print(env._task.arm.observables._observables['joints_pos']._raw_callable(env.physics)) #prints the joints states in the form of sin, cos theta
env._task.arm.configure_joints(env.physics, [0.3,0.5,0.6,0.7,1,0])
print()
print(env._task.arm.observables._observables['joints_pos']._raw_callable(env.physics)) #prints the joints states in the form of sin, cos theta
frames = []
timestep = env.reset()
while not timestep.last():
  timestep = env.step([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1])
  frames.append(timestep.observation['front_close'])
all_frames = np.concatenate(frames, axis=0)

save_video(all_frames, 'reach_vision.gif', 30)

exit()


def sample_random_action():
  return env.random_state.uniform(
      low=action_spec.minimum,
      high=action_spec.maximum,
  ).astype(action_spec.dtype, copy=False)

print(sample_random_action())

reward_spec = env.task.get_reward()
print(reward_spec)

num_episodes = 100
tau = 0.005  # Target smoothing coefficient
alpha = 0.2  # Entropy regularization coefficient
gamma = 0.99  # Discount factor
state_dim = 9
action_dim = 1
action_high = 1
action_low = -1

def create_actor_network():
    inputs = tf.keras.layers.Input(shape=(state_dim,))
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    mean = tf.keras.layers.Dense(action_dim, activation='tanh')(x)
    mean = (mean + 1.0) * (action_high - action_low) / 2.0 + action_low  # Scale to action space
    log_std = tf.keras.layers.Dense(action_dim)(x)
    return tf.keras.Model(inputs=inputs, outputs=[mean, log_std])

def create_critic_network():
    state_input = tf.keras.layers.Input(shape=(state_dim,))
    action_input = tf.keras.layers.Input(shape=(action_dim,))
    concat_input = tf.keras.layers.Concatenate()([state_input, action_input])
    x = tf.keras.layers.Dense(256, activation='relu')(concat_input)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    Q_value = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=[state_input, action_input], outputs=Q_value)


exit()
# Stepping the environment through a full episode using random actions and recording the camera
frames = []
timestep = env.reset()
frames.append(timestep.observation['front_close'])
while not timestep.last():
  timestep = env.step(sample_random_action())
  frames.append(timestep.observation['front_close'])
all_frames = np.concatenate(frames, axis=0)

save_video(all_frames, 'reach_vision.gif', 30)