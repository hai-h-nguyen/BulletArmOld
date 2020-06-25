import numpy as np
import numpy.random as npr
import pybullet as pb
from itertools import combinations

from helping_hands_rl_envs.envs.pybullet_env import NoValidPositionException

from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util

class TiltDeconstructPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(TiltDeconstructPlanner, self).__init__(env, config)
    self.objs_to_remove = []

  def getStepLeft(self):
    return 100

  def pickTallestObjOnTop(self, objects=None):
    """
    pick up the highest object that is on top
    :param objects: pool of objects
    :return: encoded action
    """
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSortedObjPoses(objects=objects)

    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2], object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, r = pose[0], pose[1], pose[2]+self.env.pick_offset, pose[5]
        if obj in self.objs_to_remove:
          self.objs_to_remove.remove(obj)
        break
      while r < 0:
        r += 2*np.pi
      while r > 2*np.pi:
        r -= 2*np.pi
    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def placeOnGround(self, padding_dist, min_dist):
    """
    place on the ground, avoiding all existing objects
    :param padding_dist: padding dist for getting valid pos
    :param min_dist: min dist to adjacent object
    :return: encoded action
    """
    existing_pos = [o.getXYPosition() for o in list(filter(lambda x: not self.isObjectHeld(x), self.env.objects))]
    try:
      place_pos = self.getValidPositions(padding_dist, min_dist, existing_pos, 1)[0]
    except NoValidPositionException:
      place_pos = self.getValidPositions(padding_dist, min_dist, [], 1)[0]
    x, y, z = place_pos[0], place_pos[1], self.env.place_offset
    y1 = np.tan(self.env.tilt_rz) * x - (self.env.workspace[0].mean() * np.tan(self.env.tilt_rz) - self.env.tilt_border)
    y2 = np.tan(self.env.tilt_rz) * x - (self.env.workspace[0].mean() * np.tan(self.env.tilt_rz) + self.env.tilt_border)
    if y > y1:
      rx = -self.env.tilt_plain_rx
      rz = self.env.tilt_rz
      d = (y - y1) * np.cos(self.env.tilt_rz)
      z += np.tan(self.env.tilt_plain_rx) * d
    elif y < y2:
      rx = -self.env.tilt_plain2_rx
      rz = self.env.tilt_rz
      d = (y2 - y) * np.cos(self.env.tilt_rz)
      z += np.tan(-self.env.tilt_plain2_rx) * d
    else:
      rx = 0
      rz = np.random.random() * np.pi * 2

    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, (rz, rx))

  def placeOnTilt(self, padding_dist, min_dist):
    existing_pos = [o.getXYPosition() for o in list(filter(lambda x: not self.isObjectHeld(x), self.env.objects))]
    while True:
      try:
        place_pos = self.getValidPositions(padding_dist, min_dist, existing_pos, 1)[0]
      except NoValidPositionException:
        place_pos = self.getValidPositions(padding_dist, min_dist, [], 1)[0]
      x, y, z = place_pos[0], place_pos[1], self.env.place_offset
      y1 = np.tan(self.env.tilt_rz) * x - (self.env.workspace[0].mean() * np.tan(self.env.tilt_rz) - self.env.tilt_border)
      y2 = np.tan(self.env.tilt_rz) * x - (self.env.workspace[0].mean() * np.tan(self.env.tilt_rz) + self.env.tilt_border)
      if y > y1:
        rx = -self.env.tilt_plain_rx
        rz = self.env.tilt_rz
        d = (y - y1) * np.cos(self.env.tilt_rz)
        z += np.tan(self.env.tilt_plain_rx) * d
        break
      elif y < y2:
        rx = -self.env.tilt_plain2_rx
        rz = self.env.tilt_rz
        d = (y2 - y) * np.cos(self.env.tilt_rz)
        z += np.tan(-self.env.tilt_plain2_rx) * d
        break
      else:
        continue
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, (rz, rx))

  def getPickingAction(self):
    if self.env.checkStructure():
      self.objs_to_remove = [o for o in self.env.structure_objs]
    if not self.objs_to_remove:
      return self.pickTallestObjOnTop()
    return self.pickTallestObjOnTop(self.objs_to_remove)

  def getPlacingAction(self):
    padding = pybullet_util.getPadding(self.getHoldingObjType(), self.env.max_block_size)
    min_distance = pybullet_util.getMinDistance(self.getHoldingObjType(), self.env.max_block_size)
    if len(self.objs_to_remove) == 0:
      return self.placeOnTilt(padding, min_distance)
    return self.placeOnGround(padding, min_distance)
