import time
from copy import deepcopy
import numpy.random as npr
import numpy as np
from itertools import combinations
from bulletarm.envs.deconstruct_envs.deconstruct_env import DeconstructEnv
from bulletarm.envs.base_env import BaseEnv
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
    self.num_class = config['num_objects'] * 2 - 1
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'max_steps' not in config:
      config['max_steps'] = 10
    super(HouseBuildingXDeconstructEnv, self).__init__(config)

  def checkStructure(self):
    ''''''
    if self.goal == '1l1l2r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
      return self._checkStack(bricks+roofs) and self._checkObjUpright(roofs[0])

    elif self.goal == '1l1b1r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
      return self._checkOnTop(blocks[0], triangles[0]) and \
          self._checkOnTop(bricks[0], blocks[0]) and \
          self._isObjOnGround(bricks[0])

    elif self.goal == '1l2b1r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      return self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            (self._checkOnTop(blocks[0], triangles[0]) or self._checkOnTop(blocks[1], triangles[0])) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1])

    elif self.goal == '1l2b2r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      return self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            self._checkOnTop(blocks[0], roofs[0]) and \
            self._checkOnTop(blocks[1], roofs[0]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
            self._checkInBetween(roofs[0], blocks[0], blocks[1])

    elif self.goal == '1l2b1l2b2r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      for i in range(len(blocks)):
        for j in range(len(blocks)):
          for k in range(len(blocks)):
            for l in range(len(bricks)):
              if i != j and j != k and k != i and \
                self._checkOnTop(bricks[l], blocks[i]) and \
                self._checkOnTop(bricks[l], blocks[j]) and \
                self._checkOnTop(blocks[i], bricks[len(bricks)-l-1]) and \
                self._checkOnTop(blocks[j], bricks[len(bricks)-l-1]) and \
                self._checkOnTop(bricks[len(bricks)-l-1], blocks[k]) and \
                self._checkOnTop(bricks[len(bricks)-l-1], blocks[6-i-j-k]) and \
                self._checkOnTop(blocks[k], roofs[0]) and \
                self._checkOnTop(blocks[6-i-j-k], roofs[0]) and \
                self._checkInBetween(bricks[l], blocks[i], blocks[j]) and \
                self._checkInBetween(bricks[len(bricks)-l-1], blocks[i], blocks[j]) and \
                self._checkInBetween(roofs[0], blocks[k], blocks[j]) and \
                self._checkInBetween(bricks[l], blocks[k], blocks[6-i-j-k]) and \
                self._checkInBetween(bricks[len(bricks)-l-1], blocks[k], blocks[6-i-j-k]) and \
                self._checkInBetween(roofs[0], blocks[k], blocks[6-i-j-k]):
                return True
      return False

    elif self.goal == '1l2b2b2r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      for i in range(len(blocks)):
        for j in range(len(blocks)):
          for k in range(len(blocks)):
            if i != j and j != k and k != i and \
              self._checkOnTop(bricks[0], blocks[i]) and \
              self._checkOnTop(bricks[0], blocks[j]) and \
              self._checkOnTop(blocks[i], blocks[k]) and \
              self._checkOnTop(blocks[j], blocks[6-i-j-k]) and \
              self._checkOnTop(blocks[k], roofs[0]) and \
              self._checkOnTop(blocks[6-i-j-k], roofs[0]) and \
              self._checkInBetween(bricks[0], blocks[i], blocks[j]) and \
              self._checkInBetween(bricks[0], blocks[k], blocks[6-i-j-k]) and \
              self._checkInBetween(roofs[0], blocks[i], blocks[j]) and \
              self._checkInBetween(roofs[0], blocks[k], blocks[6-i-j-k]):
              
                return True
      return False

    elif self.goal == '1l1l1r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
      return self._checkStack(bricks+triangles) and self._checkObjUpright(triangles[0])

  def generateStructure(self):
    if self.goal == '1l1l2r':
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

    elif self.goal == '1l2b1l2b2r':
      padding = self.max_block_size * 1.5
      pos = self.get1BaseXY(padding)
      rot = self._getValidOrientation(self.random_orientation)
      self.generateStructureShape((pos[0], pos[1], self.max_block_size/2), rot, constants.BRICK)
      
      rot_z = pb.getEulerFromQuaternion(rot)[2]
      new_pos_x_1 = pos[0] - np.sin(rot_z) * self.max_block_size*0.8
      new_pos_y_1 = pos[1] + np.cos(rot_z) * self.max_block_size*0.8
      self.generateStructureShape((new_pos_x_1, new_pos_y_1, 1.5 * self.max_block_size), rot, constants.CUBE)

      new_pos_x_2 = pos[0] + np.sin(rot_z) * self.max_block_size*0.8
      new_pos_y_2 = pos[1] - np.cos(rot_z) * self.max_block_size*0.8
      self.generateStructureShape((new_pos_x_2, new_pos_y_2, 1.5 * self.max_block_size), rot, constants.CUBE)

      self.generateStructureShape((pos[0], pos[1], 2.5*self.max_block_size), rot, constants.BRICK)
      self.generateStructureShape((new_pos_x_1, new_pos_y_1, 3.5 * self.max_block_size), rot, constants.CUBE)
      self.generateStructureShape((new_pos_x_2, new_pos_y_2, 3.5 * self.max_block_size), rot, constants.CUBE)
      self.generateStructureShape((pos[0], pos[1], 4.5*self.max_block_size), rot, constants.ROOF)

    elif self.goal == '1l2b2b2r':
      padding = self.max_block_size * 1.5
      pos = self.get1BaseXY(padding)
      rot = self._getValidOrientation(self.random_orientation)
      self.generateStructureShape((pos[0], pos[1], self.max_block_size/2), rot, constants.BRICK)
      
      rot_z = pb.getEulerFromQuaternion(rot)[2]
      new_pos_x_1 = pos[0] - np.sin(rot_z) * self.max_block_size*0.8
      new_pos_y_1 = pos[1] + np.cos(rot_z) * self.max_block_size*0.8
      self.generateStructureShape((new_pos_x_1, new_pos_y_1, 1.5 * self.max_block_size), rot, constants.CUBE)

      new_pos_x_2 = pos[0] + np.sin(rot_z) * self.max_block_size*0.8
      new_pos_y_2 = pos[1] - np.cos(rot_z) * self.max_block_size*0.8
      self.generateStructureShape((new_pos_x_2, new_pos_y_2, 1.5 * self.max_block_size), rot, constants.CUBE)

      self.generateStructureShape((new_pos_x_1, new_pos_y_1, 2.5 * self.max_block_size), rot, constants.CUBE)
      self.generateStructureShape((new_pos_x_2, new_pos_y_2, 2.5 * self.max_block_size), rot, constants.CUBE)
      self.generateStructureShape((pos[0], pos[1], 3.5*self.max_block_size), rot, constants.ROOF)

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

    self.wait(50)  

  def isSimValid(self):
    if self.goal in ['1l1l1r', '1l1b1r', '1l2b1r']:
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
      return self._checkObjUpright(triangles[0]) and DeconstructEnv.isSimValid(self)

    elif self.goal in ['1l1l2r', '1l2b2r', '1l2b2b2r', '1l2b1l2b2r']:
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
      return self._checkObjUpright(roofs[0]) and DeconstructEnv.isSimValid(self)

  def get_true_abs_state(self):
    if self.goal == '1l1b1r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      if not self._checkObjUpright(triangles[0]) or not BaseEnv.isSimValid(self):
        return self.num_class

      if self._checkOnTop(blocks[0], triangles[0]) and \
          self._checkOnTop(bricks[0], blocks[0]) and \
          self._isObjOnGround(bricks[0]):
          return 0
      
      if self._checkOnTop(bricks[0], blocks[0]) and \
          self._isObjOnGround(bricks[0]) and \
          self._isObjectHeld(triangles[0]):
          return 1

      if self._checkOnTop(bricks[0], blocks[0]) and \
          self._isObjOnGround(bricks[0]) and \
          self._isObjOnGround(triangles[0]) and \
          not triangles[0].isTouching(bricks[0]):
          return 2
      
      if self._isObjectHeld(blocks[0]) and \
          self._isObjOnGround(bricks[0]) and \
          self._isObjOnGround(triangles[0]) and \
          not triangles[0].isTouching(bricks[0]):
          return 3
      
      if self._isObjOnGround(bricks[0]) and \
          self._isObjOnGround(triangles[0]) and \
          self._isObjOnGround(blocks[0]) and \
          not triangles[0].isTouching(bricks[0]) and \
          not triangles[0].isTouching(blocks[0]) and \
          not bricks[0].isTouching(blocks[0]):
          return 4
      return self.num_class

    if self.goal == '1l2b1r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))

      if not self._checkObjUpright(triangles[0]) or not BaseEnv.isSimValid(self):
        return self.num_class

      if self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            (self._checkOnTop(blocks[0], triangles[0]) or self._checkOnTop(blocks[1], triangles[0])) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]):
            return 0

      if self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
            self._isObjectHeld(triangles[0]):
            return 1

      if self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
            not triangles[0].isTouching(bricks[0]) and \
            self._isObjOnGround(triangles[0]):
            return 2
      
      for i in range(2):
        if self._isObjectHeld(blocks[i]) and \
            self._checkOnTop(bricks[0], blocks[1-i]) and \
            not triangles[0].isTouching(bricks[0]) and \
            self._isObjOnGround(triangles[0]):
            return 3

        if self._isObjOnGround(blocks[i]) and \
            self._checkOnTop(bricks[0], blocks[1-i]) and \
            self._isObjOnGround(triangles[0]) and \
            not triangles[0].isTouching(bricks[0]) and \
            not triangles[0].isTouching(blocks[i]):
            return 4
        
        if self._isObjectHeld(blocks[i]) and \
            self._isObjOnGround(blocks[1-i]) and \
            self._isObjOnGround(triangles[0]) and \
            not triangles[0].isTouching(bricks[0]) and \
            not triangles[0].isTouching(blocks[1-i]) and \
            not bricks[0].isTouching(blocks[1-i]):
            return 5

        if self._isObjOnGround(blocks[i]) and \
            self._isObjOnGround(blocks[1-i]) and \
            self._isObjOnGround(bricks[0]) and \
            self._isObjOnGround(triangles[0]) and \
            not bricks[0].isTouching(blocks[1-i]) and \
            not bricks[0].isTouching(blocks[i]) and \
            not bricks[0].isTouching(triangles[0]) and \
            not triangles[0].isTouching(blocks[1-i]) and \
            not triangles[0].isTouching(blocks[i]):
            return 6
      return self.num_class

    if self.goal == '1l2b2r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      if not self._checkObjUpright(roofs[0]) or not BaseEnv.isSimValid(self):
        return self.num_class

      if self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            self._checkOnTop(blocks[0], roofs[0]) and \
            self._checkOnTop(blocks[1], roofs[0]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
            self._checkInBetween(roofs[0], blocks[0], blocks[1]):
        return 0

      if self._isObjectHeld(roofs[0]) and \
            self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]):
        return 1

      if self._isObjOnGround(roofs[0]) and \
            not roofs[0].isTouching(blocks[0]) and \
            not roofs[0].isTouching(blocks[1]) and \
            not roofs[0].isTouching(bricks[0]) and \
            self._checkOnTop(bricks[0], blocks[0]) and \
            self._checkOnTop(bricks[0], blocks[1]) and \
            self._checkInBetween(bricks[0], blocks[0], blocks[1]):
        return 2

      for i in range(2):
        if self._isObjectHeld(blocks[i]) and \
            self._checkOnTop(bricks[0], blocks[1-i]) and \
            self._isObjOnGround(roofs[0]) and \
            not roofs[0].isTouching(bricks[0]) and \
            not roofs[0].isTouching(blocks[1-i]):
          return 3

        if self._isObjOnGround(blocks[i]) and \
            self._isObjOnGround(roofs[0]) and \
            self._checkOnTop(bricks[0], blocks[1-i]) and \
            not roofs[0].isTouching(bricks[0]) and \
            not roofs[0].isTouching(blocks[1-i]) and \
            not roofs[0].isTouching(blocks[i]):
          return 4

        if self._isObjectHeld(blocks[i]) and \
            self._isObjOnGround(roofs[0]) and \
            self._isObjOnGround(blocks[1-i]) and \
            self._isObjOnGround(bricks[0]) and \
            not roofs[0].isTouching(bricks[0]) and \
            not roofs[0].isTouching(blocks[1-i]) and \
            not bricks[0].isTouching(blocks[1-i]):
          return 5

        if self._isObjOnGround(blocks[i]) and \
            self._isObjOnGround(blocks[1-i]) and \
            self._isObjOnGround(bricks[0]) and \
            self._isObjOnGround(roofs[0]) and \
            not roofs[0].isTouching(bricks[0]) and \
            not roofs[0].isTouching(blocks[i]) and \
            not roofs[0].isTouching(blocks[1-i]) and \
            not bricks[0].isTouching(blocks[i]) and \
            not bricks[0].isTouching(blocks[1-i]) and \
            not blocks[i].isTouching(blocks[1-i]):
          return 6
      
      return self.num_class    

    if self.goal == '1l1l1r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
         
      if not self._checkObjUpright(triangles[0]) or not BaseEnv.isSimValid(self):
        return self.num_class
      
      if self._checkStack() and self._checkObjUpright(triangles[0]):
        return 0

      if self._checkStack(bricks) and self._isObjectHeld(triangles[0]):
        return 1

      if self._checkStack(bricks) and \
          self._isObjOnGround(triangles[0]) and \
          not triangles[0].isTouching(bricks[0]) and \
          not triangles[0].isTouching(bricks[1]):
          return 2

      for i in range(2):
        if self._isObjOnGround(triangles[0]) and \
            self._isObjOnGround(bricks[i]) and \
            self._isObjectHeld(bricks[1-i]) and \
            not triangles[0].isTouching(bricks[i]):
            return 3
        
        if self._isObjOnGround(triangles[0]) and \
            self._isObjOnGround(bricks[i]) and \
            self._isObjOnGround(bricks[1-i]) and \
            not triangles[0].isTouching(bricks[i]) and \
            not triangles[0].isTouching(bricks[1-i]) and \
            not bricks[1-i].isTouching(bricks[i]):
            return 4
      return self.num_class

    if self.goal == '1l1l2r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      if not self._checkObjUpright(roofs[0]) or not BaseEnv.isSimValid(self):
        return self.num_class
      
      if self._checkStack() and self._checkObjUpright(roofs[0]):
        return 0

      if self._checkStack(bricks) and self._isObjectHeld(roofs[0]):
        return 1

      if self._checkStack(bricks) and \
          self._isObjOnGround(roofs[0]) and \
          not roofs[0].isTouching(bricks[0]) and \
          not roofs[0].isTouching(bricks[1]):
          return 2

      for i in range(2):
        if self._isObjOnGround(roofs[0]) and \
            self._isObjOnGround(bricks[i]) and \
            self._isObjectHeld(bricks[1-i]) and \
            not roofs[0].isTouching(bricks[i]):
            return 3
        
        if self._isObjOnGround(roofs[0]) and \
            self._isObjOnGround(bricks[i]) and \
            self._isObjOnGround(bricks[1-i]) and \
            not roofs[0].isTouching(bricks[i]) and \
            not roofs[0].isTouching(bricks[1-i]) and \
            not bricks[1-i].isTouching(bricks[i]):
            return 4
      return self.num_class
    
    if self.goal == '1l2b2b2r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      if not self._checkObjUpright(roofs[0]) or not BaseEnv.isSimValid(self):
        return self.num_class

      for i in range(len(blocks)):
        for j in range(len(blocks)):
          for k in range(len(blocks)):
            if i != j and j != k and k != i:
                if self._checkOnTop(bricks[0], blocks[i]) and \
                self._checkOnTop(bricks[0], blocks[j]) and \
                self._checkOnTop(blocks[i], blocks[k]) and \
                self._checkOnTop(blocks[j], blocks[6-i-j-k]) and \
                self._checkOnTop(blocks[k], roofs[0]) and \
                self._checkOnTop(blocks[6-i-j-k], roofs[0]) and \
                self._checkInBetween(bricks[0], blocks[i], blocks[j]) and \
                self._checkInBetween(bricks[0], blocks[k], blocks[6-i-j-k]) and \
                self._checkInBetween(roofs[0], blocks[i], blocks[j]) and \
                self._checkInBetween(roofs[0], blocks[k], blocks[6-i-j-k]):
                  return 0

                if self._checkOnTop(bricks[0], blocks[i]) and \
                self._checkOnTop(bricks[0], blocks[j]) and \
                self._checkOnTop(blocks[i], blocks[k]) and \
                self._checkOnTop(blocks[j], blocks[6-i-j-k]) and \
                self._checkInBetween(bricks[0], blocks[i], blocks[j]) and \
                self._checkInBetween(bricks[0], blocks[k], blocks[6-i-j-k]) and \
                self._isObjectHeld(roofs[0]):
                  return 1

                if self._checkOnTop(bricks[0], blocks[i]) and \
                self._checkOnTop(bricks[0], blocks[j]) and \
                self._checkOnTop(blocks[i], blocks[k]) and \
                self._checkOnTop(blocks[j], blocks[6-i-j-k]) and \
                self._checkInBetween(bricks[0], blocks[i], blocks[j]) and \
                self._checkInBetween(bricks[0], blocks[k], blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                not roofs[0].isTouching(bricks[0]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[k]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]):
                  return 2

                if self._checkOnTop(bricks[0], blocks[i]) and \
                self._checkOnTop(bricks[0], blocks[j]) and \
                self._checkOnTop(blocks[i], blocks[k]) and \
                self._checkInBetween(bricks[0], blocks[i], blocks[j]) and \
                self._isObjectHeld(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                not roofs[0].isTouching(bricks[0]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[k]):
                  return 3

                if self._checkOnTop(bricks[0], blocks[i]) and \
                self._checkOnTop(bricks[0], blocks[j]) and \
                self._checkOnTop(blocks[i], blocks[k]) and \
                self._checkInBetween(bricks[0], blocks[i], blocks[j]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                not roofs[0].isTouching(bricks[0]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[k]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]):
                  return 4

                if self._checkOnTop(bricks[0], blocks[i]) and \
                self._checkOnTop(bricks[0], blocks[j]) and \
                self._checkInBetween(bricks[0], blocks[i], blocks[j]) and \
                self._isObjectHeld(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                not roofs[0].isTouching(bricks[0]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]):
                  return 5

                if self._checkOnTop(bricks[0], blocks[i]) and \
                self._checkOnTop(bricks[0], blocks[j]) and \
                self._checkInBetween(bricks[0], blocks[i], blocks[j]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                not roofs[0].isTouching(bricks[0]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[k]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]):
                  return 6

                if self._checkOnTop(bricks[0], blocks[i]) and \
                self._isObjectHeld(blocks[j]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                not roofs[0].isTouching(bricks[0]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[k]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]):
                  return 7

                if self._checkOnTop(bricks[0], blocks[i]) and \
                self._isObjOnGround(blocks[j]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                not roofs[0].isTouching(bricks[0]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[k]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]):
                  return 8

                if self._isObjectHeld(blocks[i]) and \
                self._isObjOnGround(bricks[0]) and \
                self._isObjOnGround(blocks[j]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                not roofs[0].isTouching(bricks[0]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[k]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]) and \
                not bricks[0].isTouching(blocks[j]) and \
                not bricks[0].isTouching(blocks[k]) and \
                not bricks[0].isTouching(blocks[6-i-j-k]):
                  return 9

                if self._isObjOnGround(bricks[0]) and \
                self._isObjOnGround(blocks[i]) and \
                self._isObjOnGround(blocks[j]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                not roofs[0].isTouching(bricks[0]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[k]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not bricks[0].isTouching(blocks[j]) and \
                not bricks[0].isTouching(blocks[k]) and \
                not bricks[0].isTouching(blocks[6-i-j-k]):
                  return 10
      return self.num_class          

    if self.goal == '1l2b1l2b2r':
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      if not self._checkObjUpright(roofs[0]) or not BaseEnv.isSimValid(self):
        return self.num_class

      for i in range(len(blocks)):
        for j in range(len(blocks)):
          for k in range(len(blocks)):
            for l in range(len(bricks)):
              if i != j and j != k and k != i:
                if self._checkOnTop(bricks[l], blocks[i]) and \
                self._checkOnTop(bricks[l], blocks[j]) and \
                self._checkOnTop(blocks[i], bricks[len(bricks)-l-1]) and \
                self._checkOnTop(blocks[j], bricks[len(bricks)-l-1]) and \
                self._checkOnTop(bricks[len(bricks)-l-1], blocks[k]) and \
                self._checkOnTop(bricks[len(bricks)-l-1], blocks[6-i-j-k]) and \
                self._checkOnTop(blocks[k], roofs[0]) and \
                self._checkOnTop(blocks[6-i-j-k], roofs[0]) and \
                self._checkInBetween(bricks[l], blocks[i], blocks[j]) and \
                self._checkInBetween(bricks[len(bricks)-l-1], blocks[i], blocks[j]) and \
                self._checkInBetween(roofs[0], blocks[k], blocks[j]) and \
                self._checkInBetween(bricks[l], blocks[k], blocks[6-i-j-k]) and \
                self._checkInBetween(bricks[len(bricks)-l-1], blocks[k], blocks[6-i-j-k]) and \
                self._checkInBetween(roofs[0], blocks[k], blocks[6-i-j-k]):
                  return 0

                if self._checkOnTop(bricks[l], blocks[i]) and \
                self._checkOnTop(bricks[l], blocks[j]) and \
                self._checkOnTop(blocks[i], bricks[len(bricks)-l-1]) and \
                self._checkOnTop(blocks[j], bricks[len(bricks)-l-1]) and \
                self._checkOnTop(bricks[len(bricks)-l-1], blocks[k]) and \
                self._checkOnTop(bricks[len(bricks)-l-1], blocks[6-i-j-k]) and \
                self._isObjectHeld(roofs[0]) and \
                self._checkInBetween(bricks[l], blocks[i], blocks[j]) and \
                self._checkInBetween(bricks[len(bricks)-l-1], blocks[i], blocks[j]) and \
                self._checkInBetween(bricks[l], blocks[k], blocks[6-i-j-k]) and \
                self._checkInBetween(bricks[len(bricks)-l-1], blocks[k], blocks[6-i-j-k]):
                  return 1

                if self._checkOnTop(bricks[l], blocks[i]) and \
                self._checkOnTop(bricks[l], blocks[j]) and \
                self._checkOnTop(blocks[i], bricks[len(bricks)-l-1]) and \
                self._checkOnTop(blocks[j], bricks[len(bricks)-l-1]) and \
                self._checkOnTop(bricks[len(bricks)-l-1], blocks[k]) and \
                self._checkOnTop(bricks[len(bricks)-l-1], blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                self._checkInBetween(bricks[l], blocks[i], blocks[j]) and \
                self._checkInBetween(bricks[len(bricks)-l-1], blocks[i], blocks[j]) and \
                self._checkInBetween(bricks[l], blocks[k], blocks[6-i-j-k]) and \
                self._checkInBetween(bricks[len(bricks)-l-1], blocks[k], blocks[6-i-j-k]) and \
                not roofs[0].isTouching(bricks[l]) and \
                not roofs[0].isTouching(bricks[len(bricks)-l-1]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]):
                  return 2

                if self._checkOnTop(bricks[l], blocks[i]) and \
                self._checkOnTop(bricks[l], blocks[j]) and \
                self._checkOnTop(blocks[i], bricks[len(bricks)-l-1]) and \
                self._checkOnTop(blocks[j], bricks[len(bricks)-l-1]) and \
                self._checkOnTop(bricks[len(bricks)-l-1], blocks[k]) and \
                self._isObjectHeld(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                self._checkInBetween(bricks[l], blocks[i], blocks[j]) and \
                self._checkInBetween(bricks[len(bricks)-l-1], blocks[i], blocks[j]) and \
                not roofs[0].isTouching(bricks[l]) and \
                not roofs[0].isTouching(bricks[len(bricks)-l-1]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]):
                  return 3

                if self._checkOnTop(bricks[l], blocks[i]) and \
                self._checkOnTop(bricks[l], blocks[j]) and \
                self._checkOnTop(blocks[i], bricks[len(bricks)-l-1]) and \
                self._checkOnTop(blocks[j], bricks[len(bricks)-l-1]) and \
                self._checkOnTop(bricks[len(bricks)-l-1], blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                self._isObjOnGround(bricks[l]) and \
                self._checkInBetween(bricks[l], blocks[i], blocks[j]) and \
                self._checkInBetween(bricks[len(bricks)-l-1], blocks[i], blocks[j]) and \
                not roofs[0].isTouching(bricks[l]) and \
                not roofs[0].isTouching(bricks[len(bricks)-l-1]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]):
                  return 4

                if self._checkOnTop(bricks[l], blocks[i]) and \
                self._checkOnTop(bricks[l], blocks[j]) and \
                self._checkOnTop(blocks[i], bricks[len(bricks)-l-1]) and \
                self._checkOnTop(blocks[j], bricks[len(bricks)-l-1]) and \
                self._isObjectHeld(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                self._isObjOnGround(bricks[l]) and \
                self._checkInBetween(bricks[l], blocks[i], blocks[j]) and \
                self._checkInBetween(bricks[len(bricks)-l-1], blocks[i], blocks[j]) and \
                not roofs[0].isTouching(bricks[l]) and \
                not roofs[0].isTouching(bricks[len(bricks)-l-1]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]):
                  return 5

                if self._checkOnTop(bricks[l], blocks[i]) and \
                self._checkOnTop(bricks[l], blocks[j]) and \
                self._checkOnTop(blocks[i], bricks[len(bricks)-l-1]) and \
                self._checkOnTop(blocks[j], bricks[len(bricks)-l-1]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                self._isObjOnGround(bricks[l]) and \
                self._checkInBetween(bricks[l], blocks[i], blocks[j]) and \
                self._checkInBetween(bricks[len(bricks)-l-1], blocks[i], blocks[j]) and \
                not roofs[0].isTouching(bricks[l]) and \
                not roofs[0].isTouching(bricks[len(bricks)-l-1]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]) and \
                not roofs[0].isTouching(blocks[k]):
                  return 6

                if self._checkOnTop(bricks[l], blocks[i]) and \
                self._checkOnTop(bricks[l], blocks[j]) and \
                self._isObjectHeld(bricks[len(bricks)-l-1]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                self._isObjOnGround(bricks[l]) and \
                self._checkInBetween(bricks[l], blocks[i], blocks[j]) and \
                not roofs[0].isTouching(bricks[l]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]) and \
                not roofs[0].isTouching(blocks[k]):
                  return 7

                if self._checkOnTop(bricks[l], blocks[i]) and \
                self._checkOnTop(bricks[l], blocks[j]) and \
                self._isObjOnGround(bricks[len(bricks)-l-1]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                self._isObjOnGround(bricks[l]) and \
                self._checkInBetween(bricks[l], blocks[i], blocks[j]) and \
                not roofs[0].isTouching(bricks[l]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]) and \
                not roofs[0].isTouching(blocks[k]) and \
                not bricks[len(bricks)-l-1].isTouching(bricks[l]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[i]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[j]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[6-i-j-k]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[k]) and \
                not bricks[len(bricks)-l-1].isTouching(roofs[0]):
                  return 8

                if self._checkOnTop(bricks[l], blocks[i]) and \
                self._isObjectHeld(blocks[j]) and \
                self._isObjOnGround(bricks[len(bricks)-l-1]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                self._isObjOnGround(bricks[l]) and \
                not roofs[0].isTouching(bricks[l]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]) and \
                not roofs[0].isTouching(blocks[k]) and \
                not bricks[len(bricks)-l-1].isTouching(bricks[l]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[i]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[6-i-j-k]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[k]) and \
                not bricks[len(bricks)-l-1].isTouching(roofs[0]):
                  return 9

                if self._checkOnTop(bricks[l], blocks[i]) and \
                self._isObjOnGround(blocks[j]) and \
                self._isObjOnGround(bricks[len(bricks)-l-1]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                self._isObjOnGround(bricks[l]) and \
                not roofs[0].isTouching(bricks[l]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]) and \
                not roofs[0].isTouching(blocks[k]) and \
                not bricks[len(bricks)-l-1].isTouching(bricks[l]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[i]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[j]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[6-i-j-k]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[k]) and \
                not bricks[len(bricks)-l-1].isTouching(roofs[0]):
                  return 10

                if self._isObjectHeld(blocks[i]) and \
                self._isObjOnGround(blocks[j]) and \
                self._isObjOnGround(bricks[len(bricks)-l-1]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                self._isObjOnGround(bricks[l]) and \
                not roofs[0].isTouching(bricks[l]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]) and \
                not roofs[0].isTouching(blocks[k]) and \
                not bricks[len(bricks)-l-1].isTouching(bricks[l]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[j]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[6-i-j-k]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[k]) and \
                not bricks[len(bricks)-l-1].isTouching(roofs[0]) and \
                not bricks[l].isTouching(blocks[j]) and \
                not bricks[l].isTouching(blocks[6-i-j-k]) and \
                not bricks[l].isTouching(blocks[k]):
                  return 11

                if self._isObjOnGround(blocks[i]) and \
                self._isObjOnGround(blocks[j]) and \
                self._isObjOnGround(bricks[len(bricks)-l-1]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjOnGround(roofs[0]) and \
                self._isObjOnGround(bricks[l]) and \
                not roofs[0].isTouching(bricks[l]) and \
                not roofs[0].isTouching(blocks[i]) and \
                not roofs[0].isTouching(blocks[j]) and \
                not roofs[0].isTouching(blocks[6-i-j-k]) and \
                not roofs[0].isTouching(blocks[k]) and \
                not bricks[len(bricks)-l-1].isTouching(bricks[l]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[i]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[j]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[6-i-j-k]) and \
                not bricks[len(bricks)-l-1].isTouching(blocks[k]) and \
                not bricks[len(bricks)-l-1].isTouching(roofs[0]) and \
                not bricks[l].isTouching(blocks[i]) and \
                not bricks[l].isTouching(blocks[j]) and \
                not bricks[l].isTouching(blocks[6-i-j-k]) and \
                not bricks[l].isTouching(blocks[k]):
                  return 12
      return self.num_class        
    
def createHouseBuildingXDeconstructEnv(config):
  return HouseBuildingXDeconstructEnv(config)
