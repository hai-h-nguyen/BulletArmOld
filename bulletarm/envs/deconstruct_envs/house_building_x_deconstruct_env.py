import time
from copy import deepcopy
import numpy.random as npr
import numpy as np
from itertools import combinations
from bulletarm.envs.deconstruct_envs.deconstruct_env import DeconstructEnv
from bulletarm.pybullet.utils import constants
import pybullet as pb
from bulletarm.envs.utils.check_goal import CheckGoal
import random

class HouseBuildingXDeconstructEnv(DeconstructEnv):
  ''''''
  def __init__(self, config):

    self.goal = config["goal_string"]

    self.check_goal = CheckGoal(self.goal, self)
    self.check_goal.parse_goal_()

    config['num_objects'] = self.check_goal.num_objects
    super(HouseBuildingXDeconstructEnv, self).__init__(config)

  def checkStructure(self):
    ''''''
    if self.goal in ['1b1b1r', '1b1r']:
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
      return self._checkStack(blocks+triangles) and self._checkObjUpright(triangles[0])

    elif self.goal in ['1l1l2r']:
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
      return self._checkStack(bricks+roofs) and self._checkObjUpright(roofs[0])

    elif self.goal in ['2b1l1r']:
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      return self._checkOnTop(blocks[0], bricks[0]) and \
            self._checkOnTop(blocks[1], bricks[0]) and \
            self._checkOnTop(bricks[0], triangles[0]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
            self._checkInBetween(triangles[0], blocks[0], blocks[1])

    elif self.goal in ['1l1b1r']:
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
      return self._checkStack(blocks+triangles) and self._checkOnTop(bricks[0], blocks[0])

    elif self.goal in ['1l2b1r']:
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      return self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            (self._checkOnTop(blocks[0], triangles[0]) or self._checkOnTop(blocks[1], triangles[0])) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1])

    elif self.goal in ['1l2b2r']:
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      return self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            self._checkOnTop(blocks[0], roofs[0]) and \
            self._checkOnTop(blocks[1], roofs[0]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
            self._checkInBetween(roofs[0], blocks[0], blocks[1])

    elif self.goal in ['1l1l1r']:
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
      return self._checkStack(bricks+triangles) and self._checkObjUpright(triangles[0])

    elif self.goal == '2b2r':
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      return self._checkOnTop(blocks[0], roofs[0]) and \
            self._checkOnTop(blocks[1], roofs[0]) and \
            self._checkInBetween(roofs[0], blocks[0], blocks[1])

    elif self.goal == '2b1r':
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      return (self._checkOnTop(blocks[0], triangles[0]) or self._checkOnTop(blocks[1], triangles[0])) and \
            self._checkAdjacent(blocks[0], blocks[1])

    elif self.goal == '2b1b1r':
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      return (self._checkStack(blocks[1:]+triangles) or self._checkStack([blocks[0]] + blocks[2:] + triangles)) and \
            self._checkAdjacent(blocks[0], blocks[1])

    elif self.goal == '2b2b1r':
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      return self._checkAdjacent(blocks[0], blocks[1]) and \
            self._checkOnTop(blocks[0], blocks[2]) and \
            self._checkOnTop(blocks[1], blocks[3]) and \
            (self._checkOnTop(blocks[2], triangles[0]) or self._checkOnTop(blocks[3], triangles[0]))

    elif self.goal == '2b2b2r':
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      return self._checkOnTop(blocks[0], blocks[2]) and \
            self._checkOnTop(blocks[1], blocks[3]) and \
            self._checkOnTop(blocks[3], roofs[0]) and \
            self._checkOnTop(blocks[2], roofs[0]) and \
            self._checkInBetween(roofs[0], blocks[0], blocks[1])

    elif self.goal == '1l1r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      return self._checkOnTop(bricks[0], triangles[0])

    elif self.goal == '1l2r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      return self._checkOnTop(bricks[0], roofs[0])

    elif self.goal == '2b1l2r':
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      return self._checkOnTop(blocks[0], bricks[0]) and \
            self._checkOnTop(blocks[1], bricks[0]) and \
            self._checkOnTop(bricks[0], roofs[0]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
            self._checkInBetween(roofs[0], blocks[0], blocks[1])

  def generateStructure(self):
    if self.goal == '1b1r':
      padding = self.max_block_size * 1.5
      pos = self.get1BaseXY(padding)
      rot = self._getValidOrientation(self.random_orientation)
      self.generateStructureShape((pos[0], pos[1], self.max_block_size / 2), rot, constants.CUBE)
      self.generateStructureShape((pos[0], pos[1], self.max_block_size + self.max_block_size / 2),
                                  rot, constants.TRIANGLE)

    elif self.goal == '1l1l2r':
      padding = self.max_block_size * 1.5
      pos = self.get1BaseXY(padding)
      rot = self._getValidOrientation(self.random_orientation)
      for i in range(2):
        self.generateStructureShape((pos[0], pos[1], i * self.max_block_size + self.max_block_size / 2), rot,
                                    constants.BRICK)
      self.generateStructureShape((pos[0], pos[1], 2*self.max_block_size + self.max_block_size / 2),
                                  rot, constants.ROOF)

    elif self.goal == '1l1b1r':
      padding = self.max_block_size * 1.5
      pos = self.get1BaseXY(padding)
      rot = self._getValidOrientation(self.random_orientation)
      self.generateStructureShape((pos[0], pos[1], self.max_block_size / 2), rot, constants.BRICK)

      rot_z = pb.getEulerFromQuaternion(rot)[2]
      new_pos_x = pos[0] - np.sin(rot_z) * self.max_block_size*0.8
      new_pos_y = pos[1] + np.cos(rot_z) * self.max_block_size*0.8

      self.generateStructureShape((new_pos_x, new_pos_y, self.max_block_size + self.max_block_size / 2),
                                  rot, constants.CUBE)
      self.generateStructureShape((new_pos_x, new_pos_y, 2*self.max_block_size + self.max_block_size / 2),
                                  rot, constants.TRIANGLE)

    elif self.goal == '1l2b1r':
      padding = self.max_block_size * 1.5
      pos = self.get1BaseXY(padding)
      rot = self._getValidOrientation(self.random_orientation)
      self.generateStructureShape((pos[0], pos[1], self.max_block_size / 2), rot, constants.BRICK)
      rot_z = pb.getEulerFromQuaternion(rot)[2]
      new_pos_x_1 = pos[0] - np.sin(rot_z) * self.max_block_size*0.8
      new_pos_y_1 = pos[1] + np.cos(rot_z) * self.max_block_size*0.8

      self.generateStructureShape((new_pos_x_1, new_pos_y_1, 1.5 * self.max_block_size), rot, constants.CUBE)

      new_pos_x_2 = pos[0] + np.sin(rot_z) * self.max_block_size*0.8
      new_pos_y_2 = pos[1] - np.cos(rot_z) * self.max_block_size*0.8

      self.generateStructureShape((new_pos_x_2, new_pos_y_2, 1.5 * self.max_block_size), rot, constants.CUBE)

      triangle_pos = random.choice([[new_pos_x_1, new_pos_y_1], [new_pos_x_2, new_pos_y_2]])
      self.generateStructureShape((triangle_pos[0], triangle_pos[1], 2.5*self.max_block_size), rot, constants.TRIANGLE)

    elif self.goal == '1l2b2r':
      padding = self.max_block_size * 1.5
      pos = self.get1BaseXY(padding)
      rot = self._getValidOrientation(self.random_orientation)
      self.generateStructureShape((pos[0], pos[1], self.max_block_size / 2), rot, constants.BRICK)
      rot_z = pb.getEulerFromQuaternion(rot)[2]
      new_pos_x_1 = pos[0] - np.sin(rot_z) * self.max_block_size*0.8
      new_pos_y_1 = pos[1] + np.cos(rot_z) * self.max_block_size*0.8

      self.generateStructureShape((new_pos_x_1, new_pos_y_1, 1.5 * self.max_block_size), rot, constants.CUBE)

      new_pos_x_2 = pos[0] + np.sin(rot_z) * self.max_block_size*0.8
      new_pos_y_2 = pos[1] - np.cos(rot_z) * self.max_block_size*0.8

      self.generateStructureShape((new_pos_x_2, new_pos_y_2, 1.5 * self.max_block_size), rot, constants.CUBE)

      self.generateStructureShape((pos[0], pos[1], 2.5*self.max_block_size), rot, constants.ROOF)

    elif self.goal == '1l1l1r':
      padding = self.max_block_size * 1.5
      pos = self.get1BaseXY(padding)
      rot = self._getValidOrientation(self.random_orientation)
      rot_z = pb.getEulerFromQuaternion(rot)[2]
      new_pos_x = pos[0] - np.sin(rot_z) * self.max_block_size/2
      new_pos_y = pos[1] + np.cos(rot_z) * self.max_block_size/2
      for i in range(2):
        self.generateStructureShape((pos[0], pos[1], i * self.max_block_size + self.max_block_size / 2), rot,
                                    constants.BRICK)
        time.sleep(0.1)
      self.generateStructureShape((new_pos_x, new_pos_y, 2.5*self.max_block_size),
                                  rot, constants.TRIANGLE)

    elif self.goal == '1b1b1r':
      padding = self.max_block_size * 1.5
      pos = self.get1BaseXY(padding)
      rot = self._getValidOrientation(self.random_orientation)
      for i in range(2):
        self.generateStructureShape((pos[0], pos[1], i * self.max_block_size + self.max_block_size / 2), rot,
                                    constants.CUBE)
      self.generateStructureShape((pos[0], pos[1], 2*self.max_block_size + self.max_block_size / 2),
                                  rot, constants.TRIANGLE)
    elif self.goal == '2b2r':
      padding = self.max_block_size * 1.5
      min_dist = 2.1 * self.max_block_size
      max_dist = 2.2 * self.max_block_size
      pos1, pos2 = self.get2BaseXY(padding, min_dist, max_dist)
      rot1 = self._getValidOrientation(self.random_orientation)
      rot2 = self._getValidOrientation(self.random_orientation)
      self.generateStructureShape((pos1[0], pos1[1], self.max_block_size / 2), rot1, constants.CUBE)
      self.generateStructureShape((pos2[0], pos2[1], self.max_block_size / 2), rot2, constants.CUBE)

      x, y, r = self.getXYRFrom2BasePos(pos1, pos2)
      self.generateStructureShape([x, y, self.max_block_size * 1.5], pb.getQuaternionFromEuler([0., 0., r]),
                                  constants.ROOF)

    elif self.goal == '2b1b1r':
      padding = self.max_block_size * 1.5
      min_dist = 2.1 * self.max_block_size
      max_dist = 2.2 * self.max_block_size
      pos1, pos2 = self.get2BaseXY(padding, min_dist, max_dist)
      rot1 = self._getValidOrientation(self.random_orientation)
      rot2 = self._getValidOrientation(self.random_orientation)
      self.generateStructureShape((pos1[0], pos1[1], self.max_block_size / 2), rot1, constants.CUBE)
      self.generateStructureShape((pos2[0], pos2[1], self.max_block_size / 2), rot2, constants.CUBE)

      pos, rot = random.choice([[pos1, rot1], [pos2, rot2]])
      self.generateStructureShape([pos[0], pos[1], self.max_block_size * 1.5], rot, constants.CUBE)
      self.generateStructureShape([pos[0], pos[1], self.max_block_size * 2.5], rot, constants.TRIANGLE)

    elif self.goal == '2b2b1r':
      padding = self.max_block_size * 1.5
      min_dist = 2.1 * self.max_block_size
      max_dist = 2.2 * self.max_block_size
      pos1, pos2 = self.get2BaseXY(padding, min_dist, max_dist)
      rot1 = self._getValidOrientation(self.random_orientation)
      rot2 = self._getValidOrientation(self.random_orientation)

      self.generateStructureShape((pos1[0], pos1[1], self.max_block_size / 2), rot1, constants.CUBE)
      self.generateStructureShape((pos2[0], pos2[1], self.max_block_size / 2), rot2, constants.CUBE)

      self.generateStructureShape((pos1[0], pos1[1], 1.5 * self.max_block_size), rot1, constants.CUBE)
      self.generateStructureShape((pos2[0], pos2[1], 1.5 * self.max_block_size), rot2, constants.CUBE)

      pos, rot = random.choice([[pos1, rot1], [pos2, rot2]])
      self.generateStructureShape([pos[0], pos[1], self.max_block_size * 2.5], rot, constants.TRIANGLE)

    elif self.goal == '2b1r':
      padding = self.max_block_size * 1.5
      min_dist = 2.1 * self.max_block_size
      max_dist = 2.2 * self.max_block_size
      pos1, pos2 = self.get2BaseXY(padding, min_dist, max_dist)
      rot1 = self._getValidOrientation(self.random_orientation)
      rot2 = self._getValidOrientation(self.random_orientation)
      self.generateStructureShape((pos1[0], pos1[1], self.max_block_size / 2), rot1, constants.CUBE)
      self.generateStructureShape((pos2[0], pos2[1], self.max_block_size / 2), rot2, constants.CUBE)

      pos, rot = random.choice([[pos1, rot1], [pos2, rot2]])
      self.generateStructureShape([pos[0], pos[1], self.max_block_size * 1.5], rot, constants.TRIANGLE)

    elif self.goal == '2b2b2r':
      padding = self.max_block_size * 1.5
      min_dist = 2.1 * self.max_block_size
      max_dist = 2.2 * self.max_block_size
      pos1, pos2 = self.get2BaseXY(padding, min_dist, max_dist)
      rot1 = self._getValidOrientation(self.random_orientation)
      rot2 = self._getValidOrientation(self.random_orientation)

      for i in range(2):
        self.generateStructureShape((pos1[0], pos1[1], i * self.max_block_size + self.max_block_size / 2), rot1,
                                    constants.CUBE)
        self.generateStructureShape((pos2[0], pos2[1], i * self.max_block_size + self.max_block_size / 2), rot2,
                                    constants.CUBE)

      x, y, r = self.getXYRFrom2BasePos(pos1, pos2)
      self.generateStructureShape([x, y, self.max_block_size * 2.5], pb.getQuaternionFromEuler([0., 0., r]),
                                  constants.ROOF)

    elif self.goal == '1l1r':
      padding = self.max_block_size * 1.5
      pos = self.get1BaseXY(padding)
      rot = self._getValidOrientation(self.random_orientation)

      rot_z = pb.getEulerFromQuaternion(rot)[2]
      new_pos_x = pos[0] - 0.5*self.max_block_size * np.sin(rot_z)
      new_pos_y = pos[1] + 0.5*self.max_block_size * np.cos(rot_z)
    
      self.generateStructureShape((pos[0], pos[1], self.max_block_size / 2), rot, constants.BRICK)
      self.generateStructureShape((new_pos_x, new_pos_y, 1.5*self.max_block_size),
                                  rot, constants.TRIANGLE)

    elif self.goal == '1l2r':
      padding = self.max_block_size * 1.5
      pos = self.get1BaseXY(padding)
      rot = self._getValidOrientation(self.random_orientation)
      self.generateStructureShape((pos[0], pos[1], self.max_block_size / 2), rot, constants.BRICK)
      self.generateStructureShape((pos[0], pos[1], self.max_block_size + self.max_block_size / 2),
                                  rot, constants.ROOF)

    elif self.goal == '2b1l2r':
      padding = self.max_block_size * 1.5
      min_dist = 2.1 * self.max_block_size
      max_dist = 2.2 * self.max_block_size
      pos1, pos2 = self.get2BaseXY(padding, min_dist, max_dist)
      rot1 = self._getValidOrientation(self.random_orientation)
      rot2 = self._getValidOrientation(self.random_orientation)
      self.generateStructureShape((pos1[0], pos1[1], self.max_block_size / 2), rot1, constants.CUBE)
      self.generateStructureShape((pos2[0], pos2[1], self.max_block_size / 2), rot2, constants.CUBE)

      x, y, r = self.getXYRFrom2BasePos(pos1, pos2)
      self.generateStructureShape([x, y, self.max_block_size * 1.5], pb.getQuaternionFromEuler([0., 0., r]),
                                  constants.BRICK)

      self.generateStructureShape([x, y, self.max_block_size * 2.5], pb.getQuaternionFromEuler([0., 0., r]),
                                  constants.ROOF)

    elif self.goal == '2b1l1r':
      padding = self.max_block_size * 1.5
      min_dist = 2.1 * self.max_block_size
      max_dist = 2.2 * self.max_block_size
      pos1, pos2 = self.get2BaseXY(padding, min_dist, max_dist)
      rot1 = self._getValidOrientation(self.random_orientation)
      rot2 = self._getValidOrientation(self.random_orientation)
      self.generateStructureShape((pos1[0], pos1[1], self.max_block_size / 2), rot1, constants.CUBE)
      self.generateStructureShape((pos2[0], pos2[1], self.max_block_size / 2), rot2, constants.CUBE)

      x, y, r = self.getXYRFrom2BasePos(pos1, pos2)
      self.generateStructureShape([x, y, self.max_block_size * 1.5], pb.getQuaternionFromEuler([0., 0., r]),
                                  constants.BRICK)

      new_pos_x = x - 0.6*self.max_block_size * np.sin(r)
      new_pos_y = y + 0.6*self.max_block_size * np.cos(r)

      self.generateStructureShape([new_pos_x, new_pos_y, self.max_block_size * 2.5], pb.getQuaternionFromEuler([0., 0., r]),
                                  constants.TRIANGLE)

    self.wait(50)  

  def isSimValid(self):
    if self.goal in ['1l1l1r', '1l1b1r', '2b1r', '2b1b1r', '1l1r', '2b1l1r', '1l2b1r', '1b1r', '1b1b1r', '2b2b1r']:
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
      return self._checkObjUpright(triangles[0]) and DeconstructEnv.isSimValid(self)

    elif self.goal in ['2b2r', '2b2b2r', '1l2r', '1l1l2r', '2b1l2r', '1l2b2r']:
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
      return self._checkObjUpright(roofs[0]) and DeconstructEnv.isSimValid(self)

def createHouseBuildingXDeconstructEnv(config):
  return HouseBuildingXDeconstructEnv(config)
