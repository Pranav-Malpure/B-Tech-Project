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

from dm_control import manipulation
import dm_control.suite as suite


# Use svg backend for figure rendering
# config InlineBackend.figure_format = 'svg'

# # Font sizes
# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Inline video helper function
# if os.environ.get('COLAB_NOTEBOOK_TEST', False):
#   # We skip video generation during tests, as it is quite expensive.
#   display_video = lambda *args, **kwargs: None
# else:
  # def display_video(frames, framerate=30):
  #   height, width, _ = frames[0].shape
  #   dpi = 70
  #   orig_backend = matplotlib.get_backend()
  #   matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
  #   fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
  #   matplotlib.use(orig_backend)  # Switch back to the original backend.
  #   ax.set_axis_off()
  #   ax.set_aspect('equal')
  #   ax.set_position([0, 0, 1, 1])
  #   im = ax.imshow(frames[0])
  #   def update(frame):
  #     im.set_data(frame)
  #     return [im]
  #   interval = 1000/framerate
  #   anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
  #                                  interval=interval, blit=True, repeat=False)
  #   return display(HTML(anim.to_html5_video()))

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



env = manipulation.load('stack_2_of_3_bricks_random_order_vision', seed=42)
action_spec = env.action_spec()

def sample_random_action():
  return env.random_state.uniform(
      low=action_spec.minimum,
      high=action_spec.maximum,
  ).astype(action_spec.dtype, copy=False)

print(sample_random_action())
exit()
# Step the environment through a full episode using random actions and record
# the camera observations.
frames = []
timestep = env.reset()
frames.append(timestep.observation['front_close'])
while not timestep.last():
  timestep = env.step(sample_random_action())
  frames.append(timestep.observation['front_close'])
all_frames = np.concatenate(frames, axis=0)
save_video(all_frames, output_path, framerate=30)

#@title Making a video {vertical-output: true}

duration = 2    # (seconds)
framerate = 30  # (Hz)

# Visualize the joint axis
scene_option = mujoco.wrapper.core.MjvOption()
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True

# Simulate and display video.
physics = mujoco.Physics.from_xml_path('/Users/pranavmalpure/B-Tech-Project/btp/lib/python3.9/site-packages/dm_control/suite/manipulator.xml')
pixels = physics.render()
PIL.Image.fromarray(pixels)
frames = []
physics.reset()  # Reset state and time
while physics.data.time < duration:
  physics.step()
  if len(frames) < physics.data.time * framerate:
    pixels = physics.render(scene_option=scene_option)
    frames.append(pixels)
save_video(frames, 'static.gif', framerate=30)