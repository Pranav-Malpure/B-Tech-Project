__author__ = "Pranav Malpure"

"""A file derived from the file reach.py in the dm_control library, which is a task where the goal is to move the hand close to a target prop or site."""

import collections

from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.entities import props
from dm_control.manipulation.shared import arenas
from dm_control.manipulation.shared import cameras
from dm_control.manipulation.shared import constants
from dm_control.manipulation.shared import observations
from dm_control.manipulation.shared import registry
from dm_control.manipulation.shared import robots
from dm_control.manipulation.shared import tags
from dm_control.manipulation.shared import workspaces
from dm_control.composer import variation
from dm_control.utils import rewards
import numpy as np
from dm_control.entities.manipulators import kinova




_ReachWorkspace = collections.namedtuple(
    '_ReachWorkspace', ['target_bbox', 'tcp_bbox', 'arm_offset'])
global start_pos_for_print
global target_pos_for_print
global target_quat_for_print

# Ensures that the props are not touching the table before settling.
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
        lower=(-1.2, -1.2, 0.02),
        upper=(1.2, 1.2, 0.4)),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-0.2, -0.2, 0.02),
        upper=(1.2, 1.2, 0.4)),
    arm_offset=robots.ARM_OFFSET)

_SITE_WORKSPACE_2 = _ReachWorkspace(
    target_bbox=workspaces.BoundingBox(
        lower=(-0.2, -0.2, 0.02),
        upper=(1.2, 1.2, 0.4)),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-0.2, -0.2, 0.02),
        upper=(1.2, 1.2, 0.4)),
    arm_offset=robots.ARM_OFFSET)

_TARGET_RADIUS = 0.3
_TARGET_RADIUS_2 = 0.1



class Reach_task(composer.Task):
  """Bring the hand close to a target prop or site."""

  def __init__(
      self, arena, arm, hand, prop, obs_settings, workspace, control_timestep):
    """Initializes a new `Reach` task.

    Args:
      arena: `composer.Entity` instance.
      arm: `robot_base.RobotArm` instance.
      hand: `robot_base.RobotHand` instance.
      prop: `composer.Entity` instance specifying the prop to reach to, or None
        in which case the target is a fixed site whose position is specified by
        the workspace.
      obs_settings: `observations.ObservationSettings` instance.
      workspace: `_ReachWorkspace` specifying the placement of the prop and TCP.
      control_timestep: Float specifying the control timestep in seconds.
    """
    self._arena = arena
    self._arm = arm
    self._hand = hand
    self._arm.attach(self._hand)
    self._arena.attach_offset(self._arm, offset=workspace.arm_offset)
    self.control_timestep = control_timestep
    self._tcp_initializer = initializers.ToolCenterPointInitializer(
        self._hand, self._arm,
        position=distributions.Uniform(*workspace.tcp_bbox),
        quaternion=workspaces.DOWN_QUATERNION)

    # self.build_task_observables = arm._build_observables()
    # self._task_qpos_observables_array = self.build_task_observables._observables['joints_pos']._raw_callable(arm._mjcf_physics_instance)
    # self._task_qtorque_observables_array = self.build_task_observables._observables['joints_pos']._raw_callable(arm._mjcf_physics_instance)
    # obs_dict = collections.OrderedDict()
    # obs_dict['qpos'] = self._task_qpos_observables_array
    # obs_dict['qtorque'] = self._task_qtorque_observables_array
    # self._task_observables = obs_dict
    # Add custom camera observable.
    self._task_observables = cameras.add_camera_observables(
        arena, obs_settings, cameras.FRONT_CLOSE)

    target_pos_distribution = distributions.Uniform(*workspace.target_bbox)
    self._prop = prop
    if prop:
      # The prop itself is used to visualize the target location.
      self._make_target_site(parent_entity=prop, visible=False)
      self._target = self._arena.add_free_entity(prop)
      self._prop_placer = initializers.PropPlacer(
          props=[prop],
          position=target_pos_distribution,
          quaternion=workspaces.uniform_z_rotation,
          settle_physics=True)
    else:
      self._target = self._make_target_site(parent_entity=arena, visible=True)
      self._target_placer = target_pos_distribution

      obs = observable.MJCFFeature('pos', self._target)
      obs.configure(**obs_settings.prop_pose._asdict())
    #   self._task_observables['target_position'] = obs # This is for that, the artificial target site is added to the image array
    self.previous_episode_target_reached = True # Ensures that only when the target is reached we start with a new one 

    


    # Add sites for visualizing the prop and target bounding boxes.
    workspaces.add_bbox_site(
        body=self.root_entity.mjcf_model.worldbody,
        lower=workspace.tcp_bbox.lower, upper=workspace.tcp_bbox.upper,
        rgba=constants.GREEN, name='tcp_spawn_area')
    workspaces.add_bbox_site(
        body=self.root_entity.mjcf_model.worldbody,
        lower=workspace.target_bbox.lower, upper=workspace.target_bbox.upper,
        rgba=constants.BLUE, name='target_spawn_area')

  def _make_target_site(self, parent_entity, visible):
    return workspaces.add_target_site(
        body=parent_entity.mjcf_model.worldbody,
        radius=_TARGET_RADIUS, visible=visible,
        rgba=constants.RED, name='target_site')

  @property
  def root_entity(self):
    return self._arena

  @property
  def arm(self):
    return self._arm

  @property
  def hand(self):
    return self._hand
  
  @property
  def target(self):
    return self._target

  @property
  def task_observables(self):
    return self._task_observables

  def get_reward(self, physics):
    hand_pos = physics.bind(self._hand.tool_center_point).xpos
    target_pos = physics.bind(self._target).xpos
    distance = np.linalg.norm(hand_pos - target_pos)
    # print("target_pos, ", target_pos)
    # print("hand_pos ", hand_pos)
    # print("Inside Task_reach.py file, distance = ", distance)
    k = 0.0001
    # return -k*distance if distance >= _TARGET_RADIUS else 10
    if hasattr(self, 'time_inside_radius'):
      if self.time_inside_radius >= 0 and distance <= _TARGET_RADIUS:
        return 1*self.time_inside_radius*100
      elif self.time_inside_radius >= 0 and distance > _TARGET_RADIUS:
        del self.time_inside_radius
        print("HMMM")
        return -1
    else:
      normalized_distance = np.linalg.norm(2*_SITE_WORKSPACE.target_bbox.upper)
      return -np.exp(-distance/normalized_distance) 
      # return -rewards.tolerance(distance, bounds=(0, _TARGET_RADIUS), margin=_TARGET_RADIUS)

  def should_terminate_episode(self, physics):
    """Determines whether the episode should terminate given the physics state."""
    hand_pos = physics.bind(self._hand.tool_center_point).xpos
    target_pos = physics.bind(self._target).xpos
    distance = np.linalg.norm(hand_pos - target_pos)
    if distance <= _TARGET_RADIUS:
        if not hasattr(self, 'time_inside_radius'):
            self.time_inside_radius = 0  # Initialize time inside radius
        self.time_inside_radius += self.control_timestep  # Increment time inside radius
        print(self.time_inside_radius)
    # else:
    #     self.time_inside_radius = 0  # Reset time inside radius if outside the radius

    if hasattr(self, 'time_inside_radius'):
      if distance > _TARGET_RADIUS:
        raise ValueError("Something wrong")
      if self.time_inside_radius > 1:
        print("Target reached continuously for 1 seconds")
        self.previous_episode_target_reached = True
        return True
      else:
        return False
    else:
        return False

  def initialize_episode(self, physics, random_state):
    # self._hand.set_grasp(physics, close_factors=random_state.uniform())
    global start_pos_for_print
    global target_pos_for_print
    global target_quat_for_print
    print("previous_episode_target_reached?", self.previous_episode_target_reached)
    if self.previous_episode_target_reached:
      self._hand.set_grasp(physics, close_factors=1.0)
      start_pos_for_print = self._tcp_initializer(physics, random_state) #actually this function does not return this, but i changed it a bit
      if self._prop:
        self._prop_placer(physics, random_state)
      else:
        while(True):
          target_pos_for_print = self._target_placer(random_state=random_state)
          target_quat_for_print = variation.evaluate(workspaces.DOWN_QUATERNION, random_state=random_state)
          if self._tcp_initializer.check_target_feasibility(physics, random_state, target_pos_for_print, target_quat_for_print) and np.linalg.norm(start_pos_for_print - target_pos_for_print)>_TARGET_RADIUS:
            print("Target feasible, start out of target radius, so proceeding to start training/evaluation")
            break
      physics.bind(self._target).pos = target_pos_for_print
      self.previous_episode_target_reached = False
    if hasattr(self, 'time_inside_radius'):
      del self.time_inside_radius
    print("Start position:", start_pos_for_print)
    physics.bind(self._target).pos = target_pos_for_print
    print("Target position:", target_pos_for_print, target_quat_for_print)
    distance_start_target = np.linalg.norm(start_pos_for_print - target_pos_for_print)
    print("Distance from start to target:", distance_start_target)


def _reach(obs_settings, use_site):
  """Configure and instantiate a `Reach_task` task.

  Args:
    obs_settings: An `observations.ObservationSettings` instance.
    use_site: Boolean, if True then the target will be a fixed site, otherwise
      it will be a moveable Duplo brick.

  Returns:
    An instance of `reach.Reach`.
  """
  arena = arenas.Standard()
  arm = robots.make_arm_custom(obs_settings=obs_settings)
  hand = robots.make_hand_custom(obs_settings=obs_settings)

  if use_site:
    workspace = _SITE_WORKSPACE
    prop = None
  else:
    workspace = _DUPLO_WORKSPACE
    prop = props.Duplo(observable_options=observations.make_options(
        obs_settings, observations.FREEPROP_OBSERVABLES))
  task = Reach_task(arena=arena, arm=arm, hand=hand, prop=prop,
               obs_settings=obs_settings,
               workspace=workspace,
               control_timestep=constants.CONTROL_TIMESTEP)
  return task


@registry.add(tags.FEATURES, tags.EASY)
def reach_duplo_features():
  return _reach(obs_settings=observations.PERFECT_FEATURES, use_site=False)


@registry.add(tags.VISION, tags.EASY)
def reach_duplo_vision():
  return _reach(obs_settings=observations.VISION, use_site=False)


@registry.add(tags.FEATURES, tags.EASY)
def reach_site_features():
  return _reach(obs_settings=observations.PERFECT_FEATURES, use_site=True)


@registry.add(tags.VISION, tags.EASY)
def reach_site_vision():
  return _reach(obs_settings=observations.VISION, use_site=True)