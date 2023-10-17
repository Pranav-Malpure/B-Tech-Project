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
import collections

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
  
from forward_kinematics import forward_kinematics
from dm_control import composer as _composer
from agent import SACAgent


duration = 20    # (seconds)
framerate = 30  # (Hz)

import mujoco.viewer as gui_viewer
from Task_reach import Reach_task, reach_site_vision
#incomoplete task
class inverse_kinematics_task(Task):
    def __init__(self):
        pass
    
    def root_entity(self):
        pass


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

np.random.seed(42)


# Visualize the joint axis
scene_option = mujoco.wrapper.core.MjvOption()
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True

# physics = mujoco.Physics.from_xml_path('/Users/pranavmalpure/B-Tech-Project/btp/lib/python3.9/site-packages/dm_control/suite/finger.xml')
# pixels = physics.render()
# PIL.Image.fromarray(pixels)
# frames = []
# physics.reset()  # Reset state and time
# while physics.data.time < duration:
#     physics.step()
#     if len(frames) < physics.data.time * framerate:
#         pixels = physics.render(scene_option=scene_option)
#         frames.append(pixels)
# save_video(frames, 'test.gif', framerate=30)



name = 'jaco_arm'
arm = kinova.JacoArm(name)
arm._build()
arm_observables = arm._build_observables()
# print(arm)
hand = kinova.JacoHand()
# arm.attach(hand)
mjcf_physics_instance = mjcf.Physics.from_xml_path("/Users/pranavmalpure/B-Tech-Project/btp/lib/python3.9/site-packages/dm_control/third_party/kinova/jaco_arm.xml")
print(arm.observables._observables['joints_pos']._raw_callable(mjcf_physics_instance))  # get position of joints
# print(arm.observables._observables['joints_vel']._callable(mjcf_physics_instance)())
# print(arm.observables._observables['joints_pos'])


print(arm_observables.joints_pos._raw_callable(mjcf_physics_instance)) # both this and the above are alternate methods of getting the same thing
end_effector_pose = forward_kinematics(arm_observables.joints_pos._raw_callable(mjcf_physics_instance)) 
print("End effector pose is %s"%end_effector_pose[0])
print("End effector pose is %s"%end_effector_pose[1])
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

print(arm_observables.joints_pos._raw_callable(mjcf_physics_instance)) # both this and the above are alternate methods of getting the same thing
print("End effector pose now is %s"%forward_kinematics(arm_observables.joints_pos._raw_callable(mjcf_physics_instance))[0])
print("configuring now")
arm.configure_joints(mjcf_physics_instance, [1, 1, 1, 1, 1, 1])
print("End effector pose now is %s"%forward_kinematics(arm_observables.joints_pos._raw_callable(mjcf_physics_instance))[0])

mjcf_physics_instance.step()
print(arm_observables.joints_pos._raw_callable(mjcf_physics_instance)) # both this and the above are alternate methods of getting the same thing
mjcf_physics_instance.step()

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
task_object = Reach_task(arena=arena, arm=arm, hand=hand, prop=None, obs_settings=observations.VISION, workspace=_DUPLO_WORKSPACE, control_timestep=0.02)
# task_object = reach_site_vision()
env = _composer.Environment(task = task_object, time_limit = duration)
action_spec = env.action_spec()
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




exit()


while mjcf_physics_instance.data.time < duration:
    print(mjcf_physics_instance.data.time)
    arm.configure_joints(mjcf_physics_instance, [0,0,0,0,0,0])
    mjcf_physics_instance.step()
    # print(arm_observables.joints_pos._raw_callable(mjcf_physics_instance)) # both this and the above are alternate methods of getting the same thing

    # mjcf_physics_instance.step()
    # print("abcd")
    # print(arm_observables.joints_pos._raw_callable(mjcf_physics_instance)) # both this and the above are alternate methods of getting the same thing

    if len(frames) < mjcf_physics_instance.data.time * framerate:
        pixels = mjcf_physics_instance.render(scene_option=scene_option)
        frames.append(pixels)


save_video(frames, '8th_sept_evening2.gif', framerate=30)  
environment = Environment()



pixels = mjcf_physics_instance.render()
image = PIL.Image.fromarray(pixels)
image.save("7sept_image1.png")
# mjcf_physics_instance = mujoco.Physics.from_xml_path("/Users/pranavmalpure/B-Tech-Project/btp/lib/python3.9/site-packages/dm_control/third_party/kinova/jaco_arm.xml")
arm.configure_joints(mjcf_physics_instance, [0.7,0.7,0.7,0.3,0.2,0.1])
# physics_model = mujo
print()
print(arm.observables._observables['joints_pos']._raw_callable(mjcf_physics_instance))


pixels = mjcf_physics_instance.render()
image = PIL.Image.fromarray(pixels)
image.save("7sept_image2.png")

# physics = mujoco.Physics.from_xml_path(arm._mjcf_root)
pixels = mjcf_physics_instance.render()
PIL.Image.fromarray(pixels)
frames = []
mjcf_physics_instance.reset()  # Reset state and time
while mjcf_physics_instance.data.time < duration:
    mjcf_physics_instance.step()
    if len(frames) < mjcf_physics_instance.data.time * framerate:
        pixels = mjcf_physics_instance.render(scene_option=scene_option)
        frames.append(pixels)
save_video(frames, '6th_sept.gif', framerate=1000)

current_state = mjcf_physics_instance._physics_state_items()[0]
print(current_state)
exit()

pixels = physics.render()
PIL.Image.fromarray(pixels)
frames = []
physics.reset()  # Reset state and time
physics.set_control([0,0,0,0,0,0])
# physics.step(1000)
current_state = physics._physics_state_items()[0]
print(current_state)
while physics.data.time < duration:
    physics.step()
    if len(frames) < physics.data.time * framerate:
        pixels = physics.render(scene_option=scene_option)
        frames.append(pixels)


save_video(frames, 'kinova_final.gif', framerate=1000)

